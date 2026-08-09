"""
Assignment Step 2: fill in nulls in the 5 measurement columns, following
the prescribed recipe exactly, in order:

  1. Drop all rows where BOTH EngineOil columns AND all 3 Ltd columns are
     null (i.e. all 5 measurement columns null in that row — no signal
     at all to work with).
  2. Drop every EquipmentID (all of that vehicle's rows) where at least
     2 of the 3 Ltd columns are entirely null across ALL of that
     vehicle's rows — an entire-vehicle sensor outage, not fillable
     from that vehicle's own history.
  3. For remaining Ltd nulls where at least 1 Ltd value IS present in
     that row: predict the missing Ltd value(s) from the present one(s),
     using linear regression fit on globally complete-case rows (all 3
     Ltd values present). The 3 Ltd columns are all cumulative
     lifetime-to-date usage counters for the same vehicle, so they're
     strongly linearly related fleet-wide (more distance driven implies
     more fuel burned and more engine hours) — that's what makes
     cross-column prediction sensible here.
  4. For any Ltd nulls still remaining (rows where all 3 were null, so
     step 3 had nothing to predict from): linearly interpolate, per
     EquipmentID per column, "equally incremented" between the last
     non-null and next non-null value in time order. Leading/trailing
     nulls with no bounding non-null value on one side are left as NaN
     (interpolation needs both endpoints — there's nothing to
     extrapolate from safely).
  5. EngineOil nulls: where BOTH EngineOilPressure and
     EngineOilTemperature are missing, predict both from the 3 Ltd
     columns. Where only ONE is missing, predict it from the OTHER
     EngineOil column plus the 3 Ltd columns. Both use linear
     regression fit on globally complete-case rows.

Output: data/big_nss_step2_clean.parquet (+ .csv)
Run:    python assignment/step2_clean_nulls.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "big_nss.csv"
OUT_PARQUET = REPO_ROOT / "data" / "big_nss_step2_clean.parquet"
OUT_CSV = REPO_ROOT / "data" / "big_nss_step2_clean.csv"

ENGINE_OIL_COLS = ["EngineOilPressure", "EngineOilTemperature"]
LTD_COLS = ["DistanceLtd", "FuelLtd", "EngineTimeLtd"]


def fit_and_report(X: pd.DataFrame, y: pd.Series, label: str) -> LinearRegression:
    """Fit a LinearRegression on complete-case data, report held-out R²,
    then return a model refit on ALL complete-case rows (more data for
    the imputation itself than a train/test split would leave it)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    probe = LinearRegression().fit(X_train, y_train)
    r2 = r2_score(y_test, probe.predict(X_test))
    print(f"  {label}: held-out R^2 = {r2:.3f} (n={len(X)})")
    return LinearRegression().fit(X, y)


def main() -> None:
    df = pd.read_csv(SRC, dtype={"EquipmentID": str}, low_memory=False)
    df["EventTimeStamp"] = pd.to_datetime(df["EventTimeStamp"], errors="coerce")
    n0 = len(df)
    print(f"Start: {n0} rows, {df['EquipmentID'].nunique()} vehicles")

    # --- 1. Drop rows where all 5 measurement columns are null ---
    all_null = df[ENGINE_OIL_COLS + LTD_COLS].isna().all(axis=1)
    df = df.loc[~all_null].copy()
    print(f"\nStep 2.1: dropped {int(all_null.sum())} rows with all 5 measurements null "
          f"-> {len(df)} rows")

    # --- 2. Drop vehicles missing >= 2 of the 3 Ltd columns ENTIRELY ---
    ltd_all_null_per_vehicle = df.groupby("EquipmentID")[LTD_COLS].apply(lambda g: g.isna().all())
    n_missing_cols = ltd_all_null_per_vehicle.sum(axis=1)
    bad_vehicles = n_missing_cols[n_missing_cols >= 2].index
    df = df.loc[~df["EquipmentID"].isin(bad_vehicles)].copy()
    print(f"\nStep 2.2: dropped {len(bad_vehicles)} vehicles with >=2 whole Ltd columns missing "
          f"-> {len(df)} rows, {df['EquipmentID'].nunique()} vehicles")

    df = df.sort_values(["EquipmentID", "EventTimeStamp"]).reset_index(drop=True)

    # --- 3. Predict missing Ltd values from the present Ltd value(s) ---
    print("\nStep 2.3: cross-column Ltd regression imputation")
    complete_ltd = df.dropna(subset=LTD_COLS)
    for target in LTD_COLS:
        others = [c for c in LTD_COLS if c != target]
        missing_target = df[target].isna()

        # Case A: both other Ltd columns present -> 2-predictor model
        both_present = missing_target & df[others].notna().all(axis=1)
        if both_present.any():
            model = fit_and_report(
                complete_ltd[others], complete_ltd[target],
                f"{target} ~ {others[0]} + {others[1]}",
            )
            df.loc[both_present, target] = model.predict(df.loc[both_present, others])

        # Case B: only one other Ltd column present -> 1-predictor model
        for known in others:
            only_known = missing_target & df[known].notna() & df[[c for c in others if c != known][0]].isna()
            if only_known.any():
                model = fit_and_report(
                    complete_ltd[[known]], complete_ltd[target],
                    f"{target} ~ {known}",
                )
                df.loc[only_known, target] = model.predict(df.loc[only_known, [known]])

    print(f"Ltd nulls remaining after regression step: {df[LTD_COLS].isna().sum().to_dict()}")

    # --- 4. Interpolate remaining Ltd nulls per vehicle (interior gaps only) ---
    print("\nStep 2.4: per-vehicle linear interpolation for remaining Ltd nulls")
    for col in LTD_COLS:
        df[col] = df.groupby("EquipmentID")[col].transform(
            lambda s: s.interpolate(method="linear", limit_area="inside")
        )
    print(f"Ltd nulls remaining after interpolation: {df[LTD_COLS].isna().sum().to_dict()}")
    print("(any remaining have no bounding non-null on one side for that vehicle — can't interpolate)")

    # --- 5. EngineOil imputation from Ltd (+ other EngineOil) columns ---
    print("\nStep 2.5: EngineOil regression imputation")
    complete_all = df.dropna(subset=ENGINE_OIL_COLS + LTD_COLS)

    both_oil_missing = df[ENGINE_OIL_COLS].isna().all(axis=1) & df[LTD_COLS].notna().all(axis=1)
    for target in ENGINE_OIL_COLS:
        mask = both_oil_missing & df[target].isna()
        if mask.any():
            model = fit_and_report(complete_all[LTD_COLS], complete_all[target], f"{target} ~ Ltd cols (both missing)")
            df.loc[mask, target] = model.predict(df.loc[mask, LTD_COLS])

    for target in ENGINE_OIL_COLS:
        other = [c for c in ENGINE_OIL_COLS if c != target][0]
        one_missing = df[target].isna() & df[other].notna() & df[LTD_COLS].notna().all(axis=1)
        if one_missing.any():
            predictors = [other] + LTD_COLS
            model = fit_and_report(
                complete_all[predictors], complete_all[target], f"{target} ~ {other} + Ltd cols (one missing)"
            )
            df.loc[one_missing, target] = model.predict(df.loc[one_missing, predictors])

    print(f"\nEngineOil nulls remaining: {df[ENGINE_OIL_COLS].isna().sum().to_dict()}")
    print(f"Ltd nulls remaining:       {df[LTD_COLS].isna().sum().to_dict()}")

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_PARQUET}")
    print(f"Wrote {OUT_CSV}")
    print(f"\nFinal: {len(df)} rows, {df['EquipmentID'].nunique()} vehicles "
          f"(started at {n0} rows)")


if __name__ == "__main__":
    main()
