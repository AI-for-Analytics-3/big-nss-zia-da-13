"""
Derive derate-episode structure from the cleaned dataset, and measure
how much lead time actually exists before a derate onset — this grounds
the choice of prediction horizon in the data instead of picking one
blind.

`Derate` is a per-measurement STATE (confirmed from the project README),
not a one-off event flag. So a vehicle that enters derate will show a
run of consecutive Derate=True rows. The thing we actually want to
predict is the ONSET of such a run (first True after a False, or a
vehicle's very first recorded row being True).

This script, per EquipmentID (sorted by EventTimeStamp):
  1. Excludes SuspectTimestamp rows from all time-delta math (549 rows
     dataset-wide) — their timestamps aren't trustworthy, so any gap
     computed using them would be noise. They stay in the row count
     but are skipped when walking the sequence for episode/lead-time
     purposes.
  2. Tags each row with an `episode_id` (NaN if not in derate) by
     grouping consecutive Derate=True rows together.
  3. Builds an episode-level table: onset/end timestamps, row count,
     duration, and — critically — the time gap and usage gap
     (DistanceLtd / EngineTimeLtd delta) between the onset row and the
     immediately preceding non-derate row for that vehicle. That gap
     distribution is the real ceiling on how far ahead a model could
     possibly predict, since you can't predict further ahead than the
     spacing between actual measurements allows you to observe.
  4. DEBOUNCES flapping: the naive consecutive-True grouping badly
     over-counts episodes because Derate flips True/False/True within
     minutes at incident boundaries (confirmed by inspection — e.g.
     vehicle 1524 shows True, True, False, True within 90 seconds).
     74% of naive episodes were a single row. We checked the
     gap-between-naive-episodes distribution for a natural break
     between "flapping" and "genuinely separate incidents" and found
     none (it's smooth across timescales) — so DEBOUNCE_HOURS below is
     a deliberate, documented choice (1 hour) rather than a
     data-derived cutoff: any two naive episodes on the same vehicle
     less than DEBOUNCE_HOURS apart are merged into one incident.

Outputs:
  data/derate_episodes_naive.csv — one row per naive (un-debounced) consecutive-True run
  data/derate_episodes.csv       — one row per DEBOUNCED incident (the one to use downstream)
  Printed summary of lead-time / sampling-interval distributions.

Run:
    python scripts/build_derate_episodes.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_DIR / "big_nss_clean.parquet"
OUT_EPISODES_NAIVE = DATA_DIR / "derate_episodes_naive.csv"
OUT_EPISODES = DATA_DIR / "derate_episodes.csv"

# Two naive episodes (consecutive-True runs) on the same vehicle separated
# by less than this many hours are merged into a single incident. See
# module docstring point 4 — chosen deliberately, no clean data-derived
# cutoff exists; tune here if a different debounce window is wanted.
DEBOUNCE_HOURS = 1.0


def main() -> None:
    df = pd.read_parquet(SRC)

    # Drop suspect-timestamp rows for this sequence-walking analysis —
    # keep the rest of the pipeline (cleaned dataset) untouched.
    n_total = len(df)
    df = df.loc[~df["SuspectTimestamp"]].copy()
    n_suspect_dropped = n_total - len(df)
    print(f"Excluding {n_suspect_dropped} SuspectTimestamp rows from episode/lead-time analysis "
          f"({len(df)} rows remain)")

    df = df.sort_values(["EquipmentID", "EventTimeStamp"]).reset_index(drop=True)

    # --- Row-to-row sampling interval per vehicle (context for horizon choice) ---
    df["prev_ts_same_veh"] = df.groupby("EquipmentID")["EventTimeStamp"].shift(1)
    df["gap_hours"] = (df["EventTimeStamp"] - df["prev_ts_same_veh"]).dt.total_seconds() / 3600.0
    gap_stats = df["gap_hours"].dropna()
    print("\n--- Row-to-row time gap per vehicle (all rows, not just onsets) ---")
    print(gap_stats.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    # --- Episode tagging: consecutive Derate=True rows per vehicle ---
    df["prev_derate_same_veh"] = df.groupby("EquipmentID")["Derate"].shift(1)
    df["is_onset"] = df["Derate"] & (df["prev_derate_same_veh"].fillna(False) == False)
    df["episode_id"] = np.where(df["Derate"], df.groupby("EquipmentID")["is_onset"].cumsum(), np.nan)
    # Make episode_id globally unique (equipment + local episode number)
    df["episode_key"] = np.where(
        df["Derate"],
        df["EquipmentID"] + "_" + df["episode_id"].astype("Int64").astype(str),
        None,
    )

    n_episodes = int(df["is_onset"].sum())
    n_vehicles_with_derate = df.loc[df["Derate"], "EquipmentID"].nunique()
    print(f"\nDerate rows: {int(df['Derate'].sum())} across {n_episodes} episodes, "
          f"{n_vehicles_with_derate} distinct vehicles")

    # --- Lead time before onset: gap to the last non-derate row for that vehicle ---
    df["prev_DistanceLtd"] = df.groupby("EquipmentID")["DistanceLtd"].shift(1)
    df["prev_EngineTimeLtd"] = df.groupby("EquipmentID")["EngineTimeLtd"].shift(1)

    onset_rows = df.loc[df["is_onset"]].copy()
    onset_rows = onset_rows.rename(columns={"gap_hours": "lead_time_hours"})
    onset_rows["lead_distance"] = onset_rows["DistanceLtd"] - onset_rows["prev_DistanceLtd"]
    onset_rows["lead_engine_time"] = onset_rows["EngineTimeLtd"] - onset_rows["prev_EngineTimeLtd"]

    print("\n--- Lead time (hours) between onset row and the prior row, same vehicle ---")
    print(onset_rows["lead_time_hours"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    print("\n--- Lead usage-gap: DistanceLtd delta into onset ---")
    print(onset_rows["lead_distance"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    print("\n--- Lead usage-gap: EngineTimeLtd delta into onset ---")
    print(onset_rows["lead_engine_time"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    # How many onsets are a vehicle's very first recorded row (no lead time possible at all)?
    n_first_row_onsets = int(onset_rows["lead_time_hours"].isna().sum())
    print(f"\nOnsets with no prior row at all for that vehicle (first record is already in derate): "
          f"{n_first_row_onsets} / {len(onset_rows)}")

    # --- Episode-level table: onset info + episode end/duration ---
    derate_rows = df.loc[df["Derate"]].copy()
    ep_end = derate_rows.groupby("episode_key")["EventTimeStamp"].max().rename("end_ts")
    ep_n = derate_rows.groupby("episode_key").size().rename("n_rows")

    episodes = onset_rows.set_index("episode_key")[
        ["EquipmentID", "EventTimeStamp", "lead_time_hours", "lead_distance", "lead_engine_time"]
    ].rename(columns={"EventTimeStamp": "onset_ts"}).join([ep_end, ep_n])
    episodes["duration_hours"] = (episodes["end_ts"] - episodes["onset_ts"]).dt.total_seconds() / 3600.0
    episodes = episodes.reset_index()

    episodes.to_csv(OUT_EPISODES_NAIVE, index=False)
    print(f"\nWrote {OUT_EPISODES_NAIVE} ({len(episodes)} naive episodes, pre-debounce)")

    print("\n--- Naive episode duration (hours) ---")
    print(episodes["duration_hours"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
    print(f"Single-row naive episodes: {int((episodes['n_rows'] == 1).sum())} / {len(episodes)}")

    # --- Debounce: merge naive episodes on the same vehicle that are
    # closer together than DEBOUNCE_HOURS into one incident ---
    episodes = episodes.sort_values(["EquipmentID", "onset_ts"]).reset_index(drop=True)
    episodes["prev_end_same_veh"] = episodes.groupby("EquipmentID")["end_ts"].shift(1)
    episodes["gap_to_prev_hours"] = (
        episodes["onset_ts"] - episodes["prev_end_same_veh"]
    ).dt.total_seconds() / 3600.0
    # Start a new incident whenever there's no previous naive episode for
    # this vehicle, or the gap exceeds the debounce window.
    episodes["new_incident"] = episodes["gap_to_prev_hours"].isna() | (
        episodes["gap_to_prev_hours"] >= DEBOUNCE_HOURS
    )
    episodes["incident_id"] = episodes.groupby("EquipmentID")["new_incident"].cumsum()
    episodes["incident_key"] = episodes["EquipmentID"] + "_" + episodes["incident_id"].astype(str)

    incidents = episodes.groupby("incident_key").agg(
        EquipmentID=("EquipmentID", "first"),
        onset_ts=("onset_ts", "min"),
        end_ts=("end_ts", "max"),
        n_naive_episodes=("episode_key", "size"),
        n_rows=("n_rows", "sum"),
        # Lead time/usage-gap belong to the FIRST naive episode in the
        # incident — later sub-episodes are flapping, not new onsets.
        lead_time_hours=("lead_time_hours", "first"),
        lead_distance=("lead_distance", "first"),
        lead_engine_time=("lead_engine_time", "first"),
    ).reset_index()
    incidents["duration_hours"] = (incidents["end_ts"] - incidents["onset_ts"]).dt.total_seconds() / 3600.0

    incidents.to_csv(OUT_EPISODES, index=False)
    print(f"\nWrote {OUT_EPISODES} ({len(incidents)} debounced incidents, "
          f"DEBOUNCE_HOURS={DEBOUNCE_HOURS}, collapsed from {len(episodes)} naive episodes)")

    print("\n--- Debounced incident duration (hours) ---")
    print(incidents["duration_hours"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
    print(f"Single-naive-episode incidents: {int((incidents['n_naive_episodes'] == 1).sum())} / {len(incidents)}")

    print("\n--- Incidents per vehicle ---")
    print(incidents.groupby("EquipmentID").size().describe())


if __name__ == "__main__":
    main()
