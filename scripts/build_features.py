"""
Feature engineering for the derate-prediction model.

Builds, per EquipmentID (time-sorted), history/trend features from the
telemetry sensors, then joins them onto the row-level target table from
build_target.py.

Design notes (per user decisions):

- Multiple lookback windows: short (last 5 readings, row-count based —
  robust to irregular sampling), medium (trailing 24h), long (trailing 7d).
- Missing sensors (structural ~50% missingness) are forward-filled per
  vehicle for "current known value" features, paired with
  HoursSinceLastTelemetry so the model knows how stale that value is.
- IMPORTANT: rolling mean/std/min/max are computed on the RAW (non-ffilled)
  sensor values, not the ffilled series. Rolling over ffilled data would
  repeat the same value across many duplicated rows within a window,
  artificially shrinking rolling std and biasing the mean toward
  whatever value happened to persist longest — ffill is only appropriate
  for a single "current best known value" feature, not for windowed
  statistics. Pandas rolling functions already skip NaN correctly
  (count/mean over whatever valid values fall in the window), so this
  is both simpler and more statistically honest.

Two families of sensor:
  - "Point condition" (EngineOilPressure, EngineOilTemperature): rolling
    mean/std per window, plus deviation-from-recent-baseline (current
    ffilled value minus medium/long rolling mean) as a spike indicator.
  - "Cumulative usage" (DistanceLtd, FuelLtd, EngineTimeLtd): monotonic
    lifetime-to-date counters, so windowed rolling mean is not
    meaningful — instead we take rolling max-min within each raw-value
    window as "usage accumulated in that trailing window" (recent
    intensity of use), plus the ffilled current cumulative value as a
    vehicle-age/wear baseline feature.

Also adds recurrence-risk features from the debounced incident table:
  - PriorIncidentCount: how many derate incidents this vehicle has
    already had before this row
  - HoursSinceLastIncident: hours since this vehicle's last incident
    ended (NaN if none yet)

Output: data/model_ready.parquet
Run:    python scripts/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC_CLEAN = DATA_DIR / "big_nss_clean.parquet"
SRC_TARGET = DATA_DIR / "training_rows.parquet"
SRC_INCIDENTS = DATA_DIR / "derate_episodes.csv"
OUT = DATA_DIR / "model_ready.parquet"

POINT_SENSORS = ["EngineOilPressure", "EngineOilTemperature"]
CUMULATIVE_SENSORS = ["DistanceLtd", "FuelLtd", "EngineTimeLtd"]
SHORT_WINDOW_ROWS = 5


def _rolling_rows(df: pd.DataFrame, col: str, window, stat: str, time_based: bool):
    """Row-aligned rolling stat. df must already be sorted by
    [EquipmentID, EventTimeStamp] so group blocks and within-group order
    match the frame's row order — that lets us assign the result back
    positionally via .to_numpy(), sidestepping the fact that pandas'
    time-based groupby-rolling indexes results by (EquipmentID,
    EventTimeStamp), which collides whenever a vehicle has duplicate
    timestamps (confirmed present in this data)."""
    g = df.groupby("EquipmentID", sort=False)
    if time_based:
        roll = g.rolling(window, on="EventTimeStamp", min_periods=1)[col]
    else:
        roll = g[col].rolling(window, min_periods=1)
    return getattr(roll, stat)().to_numpy()


def add_point_sensor_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_ffill"] = df.groupby("EquipmentID", sort=False)[col].ffill()

    df[f"{col}_short_mean"] = _rolling_rows(df, col, SHORT_WINDOW_ROWS, "mean", time_based=False)
    df[f"{col}_short_std"] = _rolling_rows(df, col, SHORT_WINDOW_ROWS, "std", time_based=False)

    df[f"{col}_medium_mean"] = _rolling_rows(df, col, "24h", "mean", time_based=True)
    df[f"{col}_medium_std"] = _rolling_rows(df, col, "24h", "std", time_based=True)

    df[f"{col}_long_mean"] = _rolling_rows(df, col, "7D", "mean", time_based=True)
    df[f"{col}_long_std"] = _rolling_rows(df, col, "7D", "std", time_based=True)

    df[f"{col}_dev_medium"] = df[f"{col}_ffill"] - df[f"{col}_medium_mean"]
    df[f"{col}_dev_long"] = df[f"{col}_ffill"] - df[f"{col}_long_mean"]
    return df


def add_cumulative_sensor_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_ffill"] = df.groupby("EquipmentID", sort=False)[col].ffill()

    df[f"{col}_short_delta"] = (
        _rolling_rows(df, col, SHORT_WINDOW_ROWS, "max", time_based=False)
        - _rolling_rows(df, col, SHORT_WINDOW_ROWS, "min", time_based=False)
    )
    df[f"{col}_medium_delta"] = (
        _rolling_rows(df, col, "24h", "max", time_based=True)
        - _rolling_rows(df, col, "24h", "min", time_based=True)
    )
    df[f"{col}_long_delta"] = (
        _rolling_rows(df, col, "7D", "max", time_based=True)
        - _rolling_rows(df, col, "7D", "min", time_based=True)
    )
    return df


def main() -> None:
    df = pd.read_parquet(SRC_CLEAN)
    df = df.loc[~df["SuspectTimestamp"]].copy()
    df = df.sort_values(["EquipmentID", "EventTimeStamp"]).reset_index(drop=True)

    print(f"Building features over {len(df)} rows / {df['EquipmentID'].nunique()} vehicles ...")

    # --- Staleness: hours since the last row with any telemetry present ---
    df["_last_telemetry_ts"] = df["EventTimeStamp"].where(df["HasTelemetry"])
    df["_last_telemetry_ts"] = df.groupby("EquipmentID", sort=False)["_last_telemetry_ts"].ffill()
    df["HoursSinceLastTelemetry"] = (
        (df["EventTimeStamp"] - df["_last_telemetry_ts"]).dt.total_seconds() / 3600.0
    )
    df = df.drop(columns=["_last_telemetry_ts"])

    for col in POINT_SENSORS:
        print(f"  point-sensor features: {col}")
        df = add_point_sensor_features(df, col)

    for col in CUMULATIVE_SENSORS:
        print(f"  cumulative-sensor features: {col}")
        df = add_cumulative_sensor_features(df, col)

    # --- Recurrence-risk features from the debounced incident table ---
    incidents = pd.read_csv(SRC_INCIDENTS, dtype={"EquipmentID": str})
    incidents["onset_ts"] = pd.to_datetime(incidents["onset_ts"])
    incidents["end_ts"] = pd.to_datetime(incidents["end_ts"])
    incidents = incidents.sort_values(["EquipmentID", "onset_ts"])

    prior_incident_count = np.zeros(len(df), dtype=int)
    hours_since_last_incident = np.full(len(df), np.nan)

    ends_by_veh = {veh: np.sort(g["end_ts"].values) for veh, g in incidents.groupby("EquipmentID")}
    for veh, idx in df.groupby("EquipmentID", sort=False).indices.items():
        ends = ends_by_veh.get(veh)
        if ends is None:
            continue
        row_ts = df["EventTimeStamp"].values[idx]
        pos = np.searchsorted(ends, row_ts, side="right")  # # of incidents fully ended by this row
        prior_incident_count[idx] = pos
        has_prior = pos > 0
        if has_prior.any():
            last_end = ends[pos[has_prior] - 1]
            gap_ns = (row_ts[has_prior] - last_end).astype("timedelta64[ns]").astype(np.int64)
            tmp = hours_since_last_incident[idx]
            tmp[has_prior] = gap_ns / 3.6e12
            hours_since_last_incident[idx] = tmp

    df["PriorIncidentCount"] = prior_incident_count
    df["HoursSinceLastIncident"] = hours_since_last_incident

    # --- Join engineered features onto the target/candidate rows ---
    targets = pd.read_parquet(SRC_TARGET)
    feature_cols = [c for c in df.columns if c not in targets.columns or c == "RecordID"]
    model_ready = targets.merge(df[["RecordID"] + [c for c in feature_cols if c != "RecordID"]],
                                 on="RecordID", how="left")

    model_ready.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({model_ready.shape[0]} rows, {model_ready.shape[1]} columns)")
    print("\nFeature columns added:")
    for c in feature_cols:
        if c != "RecordID":
            print(f"  {c}")


if __name__ == "__main__":
    main()
