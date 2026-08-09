"""
Clean the Big NSS Express telemetry/derate dataset.

Input:  data/big_nss.csv
Output: data/big_nss_clean.parquet (+ data/big_nss_clean.csv)

What this does, and why:

1. Drops the stray `Unnamed: 0` column — it's just a leftover pandas
   row-index written out by a prior `to_csv()` call and carries no
   information.

2. Fixes 3 corrupted EquipmentID values (`R1762`, `R1764`, `2185A`).
   Each has a clean numeric twin already present in the data
   (1762, 1764, 2185 respectively) with far more rows, so these look
   like typos/OCR-style noise on an existing ID rather than new
   vehicles. We strip the stray leading/trailing letters and merge
   them into the existing numeric ID. EquipmentID is kept as a string
   dtype throughout (it's a categorical label, not a quantity).

3. Flags (does NOT drop) 549 rows whose EventTimeStamp falls way
   outside the dataset's core range (2015-01-01 through 2020-12-31 —
   the bulk of the 1.19M records). These stray timestamps land in
   2000, 2002, 2009-2011, and 2026, which is implausible for this
   fleet's telemetry window. Because 5 of these rows are rare
   Derate=True positives (only ~1,195 positives exist in the whole
   dataset), we do not want to silently delete them. Instead we add
   a `SuspectTimestamp` flag so downstream time-series feature work
   (e.g. "time since last event per vehicle") can exclude them
   without losing the label information entirely.

4. Adds a `HasTelemetry` flag rather than imputing the sensor columns
   (EngineOilPressure, EngineOilTemperature, DistanceLtd, FuelLtd,
   EngineTimeLtd). ~599,700 rows (50.5%) have ALL five sensor columns
   null together, which is a structural pattern, not random
   missingness — it looks like this table merges two source record
   types (fault/derate events vs. diagnostic telemetry snapshots).
   Derate=True is nearly evenly split across both groups (631 vs 564),
   so telemetry presence is not itself predictive of the label, and
   mean/median imputation across record types would blend two
   different populations. A further ~10,837 rows have partial
   nulls (some sensor columns present, others not) — these are left
   as-is (NaN) rather than imputed, since we don't yet know the cause.

5. Sorts by EquipmentID then EventTimeStamp, which matters for any
   per-vehicle time-series / lag feature engineering later.

Run:
    python scripts/clean_data.py
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_DIR / "big_nss.csv"
OUT_PARQUET = DATA_DIR / "big_nss_clean.parquet"
OUT_CSV = DATA_DIR / "big_nss_clean.csv"

SENSOR_COLS = [
    "EngineOilPressure",
    "EngineOilTemperature",
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]

# Core date range the vast majority of records fall in; anything
# outside this is flagged as suspect rather than dropped.
CORE_START = pd.Timestamp("2014-01-01")
CORE_END = pd.Timestamp("2020-12-31 23:59:59")


def clean_equipment_id(raw: pd.Series) -> pd.Series:
    """Strip stray non-digit characters from otherwise-numeric IDs."""
    cleaned = raw.str.strip()
    stripped = cleaned.apply(lambda v: re.sub(r"[^0-9]", "", v) if re.search(r"[^0-9]", v) else v)
    return stripped


def main() -> None:
    print(f"Reading {SRC} ...")
    df = pd.read_csv(SRC, dtype={"EquipmentID": str}, low_memory=False)
    n_start = len(df)

    # 1. Drop artifact index column.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # 2. Fix corrupted EquipmentID values.
    bad_mask = df["EquipmentID"].str.contains(r"[^0-9]", regex=True, na=False)
    n_bad_ids = int(bad_mask.sum())
    df["EquipmentID"] = clean_equipment_id(df["EquipmentID"])

    # 3. Parse timestamps, flag out-of-core-range rows instead of dropping.
    df["EventTimeStamp"] = pd.to_datetime(df["EventTimeStamp"], errors="coerce")
    df["SuspectTimestamp"] = ~df["EventTimeStamp"].between(CORE_START, CORE_END)
    n_suspect_ts = int(df["SuspectTimestamp"].sum())

    # 4. Structural-missingness flag for the sensor block.
    df["HasTelemetry"] = df[SENSOR_COLS].notna().any(axis=1)
    n_no_telemetry = int((~df["HasTelemetry"]).sum())

    # 5. Sort for downstream per-vehicle time-series work.
    df = df.sort_values(["EquipmentID", "EventTimeStamp"]).reset_index(drop=True)

    n_end = len(df)
    assert n_end == n_start, "Row count changed unexpectedly during cleaning"

    print(f"Rows: {n_start} (unchanged, no rows dropped)")
    print(f"EquipmentID values fixed: {n_bad_ids}")
    print(f"Rows flagged SuspectTimestamp (outside {CORE_START.date()}..{CORE_END.date()}): {n_suspect_ts}")
    print(f"Rows flagged HasTelemetry=False (no sensor data at all): {n_no_telemetry}")
    print(f"Derate positives total: {int(df['Derate'].sum())}")
    print(f"  of which SuspectTimestamp: {int(df.loc[df['Derate'], 'SuspectTimestamp'].sum())}")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_PARQUET}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
