"""
Build the row-level prediction target: for each candidate row, will this
vehicle enter a full-derate incident within the next HORIZON_HOURS?

Uses the debounced incident table from build_derate_episodes.py (each row
= one real derate incident onset, flapping already merged) rather than
raw Derate rows, so the label reflects genuine incidents.

Candidate rows = all rows where:
  - Derate == False (a row already in derate isn't a "predict ahead" case)
  - SuspectTimestamp == False (549 rows with untrustworthy timestamps
    dataset-wide; can't compute a reliable time-to-next-incident for them)

For each candidate row, per vehicle, find the next incident onset at or
after its timestamp and compute the gap in hours. Target is 1 if that
gap is <= HORIZON_HOURS.

HORIZON_HOURS = 6 is a starting default grounded in the lead-time
analysis (build_derate_episodes.py): 75th percentile lead time into an
onset is ~4h, 90th is ~15.7h, so 6h sits just past the bulk of the
distribution — long enough to be operationally actionable, short enough
that most positive labels still have real preceding signal to learn
from. Tune the constant and re-run to compare horizons.

Output: data/training_rows.parquet
Run:    python scripts/build_target.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC_CLEAN = DATA_DIR / "big_nss_clean.parquet"
SRC_INCIDENTS = DATA_DIR / "derate_episodes.csv"
OUT = DATA_DIR / "training_rows.parquet"

HORIZON_HOURS = 6.0


def main() -> None:
    df = pd.read_parquet(SRC_CLEAN)
    incidents = pd.read_csv(SRC_INCIDENTS, dtype={"EquipmentID": str})
    incidents["onset_ts"] = pd.to_datetime(incidents["onset_ts"])

    n_total = len(df)
    candidates = df.loc[(~df["Derate"]) & (~df["SuspectTimestamp"])].copy()
    print(f"Candidate rows: {len(candidates)} / {n_total} "
          f"(excluded {int(df['Derate'].sum())} in-derate rows and "
          f"{int(df['SuspectTimestamp'].sum())} suspect-timestamp rows, "
          f"with some overlap between the two)")

    candidates = candidates.sort_values(["EquipmentID", "EventTimeStamp"]).reset_index(drop=True)
    onsets_by_veh = {
        veh: np.sort(group["onset_ts"].values)
        for veh, group in incidents.groupby("EquipmentID")
    }

    time_to_next = np.full(len(candidates), np.nan)
    veh_col = candidates["EquipmentID"].values
    ts_col = candidates["EventTimeStamp"].values

    for veh, group_idx in candidates.groupby("EquipmentID").indices.items():
        onset_arr = onsets_by_veh.get(veh)
        if onset_arr is None:
            continue
        row_ts = ts_col[group_idx]
        # searchsorted 'left' gives the index of the first onset >= row_ts
        pos = np.searchsorted(onset_arr, row_ts, side="left")
        has_future = pos < len(onset_arr)
        gap_hours = np.full(len(group_idx), np.nan)
        if has_future.any():
            next_onset = onset_arr[pos[has_future]]
            gap_ns = (next_onset - row_ts[has_future]).astype("timedelta64[ns]").astype(np.int64)
            gap_hours[has_future] = gap_ns / 3.6e12
        time_to_next[group_idx] = gap_hours

    candidates["TimeToNextIncident_hours"] = time_to_next
    candidates["DerateWithinHorizon"] = (
        candidates["TimeToNextIncident_hours"] <= HORIZON_HOURS
    ) & candidates["TimeToNextIncident_hours"].notna()

    n_pos = int(candidates["DerateWithinHorizon"].sum())
    n_rows = len(candidates)
    print(f"\nHORIZON_HOURS = {HORIZON_HOURS}")
    print(f"Positive rows (DerateWithinHorizon=True): {n_pos} / {n_rows} "
          f"({100 * n_pos / n_rows:.3f}%)")

    n_vehicles_with_positive = candidates.loc[candidates["DerateWithinHorizon"], "EquipmentID"].nunique()
    print(f"Distinct vehicles contributing a positive row: {n_vehicles_with_positive}")

    print("\n--- HasTelemetry among positive vs negative rows ---")
    print(candidates.groupby("DerateWithinHorizon")["HasTelemetry"].mean())

    candidates.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(candidates)} rows, {candidates.shape[1]} columns)")


if __name__ == "__main__":
    main()
