"""
Assignment Step 4:
Predict whether a vehicle will experience a future derate.

Prediction horizons:
- 12 hours
- 1 day
- 3 days
- 7 days
- 14 days
- 21 days

The model uses the current values of:
- EngineOilPressure
- EngineOilTemperature
- DistanceLtd
- FuelLtd
- EngineTimeLtd

Evaluation:
ROC AUC

Run:
py -3.13 big-nss-zia-da-13\\assignment\\step4_future_derate.py
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC = REPO_ROOT / "data" / "big_nss_step2_clean.parquet"

OUT_RESULTS = (
    REPO_ROOT
    / "assignment"
    / "outputs"
    / "step4_roc_auc_results.csv"
)


# ---------------------------------------------------------
# Model features
# ---------------------------------------------------------

FEATURES = [
    "EngineOilPressure",
    "EngineOilTemperature",
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]


# ---------------------------------------------------------
# Prediction horizons
# ---------------------------------------------------------

HORIZONS = {
    "12 hours": pd.Timedelta(hours=12),
    "1 day": pd.Timedelta(days=1),
    "3 days": pd.Timedelta(days=3),
    "7 days": pd.Timedelta(days=7),
    "14 days": pd.Timedelta(days=14),
    "21 days": pd.Timedelta(days=21),
}


# ---------------------------------------------------------
# Create future derate target
# ---------------------------------------------------------

def create_future_target(
    data: pd.DataFrame,
    horizon: pd.Timedelta
) -> pd.Series:

    """
    For every row, determine whether the same vehicle
    experiences a derate AFTER the current timestamp
    and within the requested future horizon.

    Uses pandas merge_asof rather than iterrows.
    """

    # Get only actual derate events
    derate_events = (
        data.loc[
            data["Derate"] == True,
            ["EquipmentID", "EventTimeStamp"]
        ]
        .dropna()
        .sort_values(["EventTimeStamp", "EquipmentID"])
        .rename(
            columns={
                "EventTimeStamp": "NextDerateTime"
            }
        )
    )

    # Rows for which we want a future target
    current_rows = (
        data[
            ["EquipmentID", "EventTimeStamp"]
        ]
        .copy()
    )

    current_rows["_row_id"] = range(len(current_rows))

    current_rows = current_rows.sort_values(
        ["EventTimeStamp", "EquipmentID"]
    )


    # Find the next derate for the same EquipmentID
    matched = pd.merge_asof(
        current_rows,
        derate_events,
        left_on="EventTimeStamp",
        right_on="NextDerateTime",
        by="EquipmentID",
        direction="forward",
        allow_exact_matches=False,
    )


    # Calculate time until the next derate
    time_until_derate = (
        matched["NextDerateTime"]
        - matched["EventTimeStamp"]
    )


    # True when the next derate occurs inside the horizon
    matched["FutureDerate"] = (
        time_until_derate.notna()
        & (time_until_derate <= horizon)
    )


    # Restore original row order
    matched = matched.sort_values("_row_id")

    return (
        matched["FutureDerate"]
        .astype(int)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("==================================================")
    print("STEP 4: FUTURE DERATE PREDICTION")
    print("==================================================")


    # -----------------------------------------------------
    # Read cleaned data
    # -----------------------------------------------------

    big_nss = pd.read_parquet(SRC)


    big_nss["EventTimeStamp"] = pd.to_datetime(
        big_nss["EventTimeStamp"],
        errors="coerce"
    )


    # Remove the few remaining missing measurements
    model_data = (
        big_nss
        .dropna(
            subset=
            FEATURES
            + [
                "EquipmentID",
                "EventTimeStamp",
                "Derate",
            ]
        )
        .copy()
    )


    # Sort for time-based operations
    model_data = model_data.sort_values(
        ["EquipmentID", "EventTimeStamp"]
    ).reset_index(drop=True)


    print(f"\nRows available: {len(model_data)}")

    print(
        "Vehicles:",
        model_data["EquipmentID"].nunique()
    )


    # -----------------------------------------------------
    # X and vehicle groups
    # -----------------------------------------------------

    X = model_data[FEATURES]

    groups = model_data["EquipmentID"]


    # -----------------------------------------------------
    # Split trucks into train and test groups
    # -----------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )


    train_index, test_index = next(
        splitter.split(
            X,
            groups=groups
        )
    )


    X_train = X.iloc[train_index]

    X_test = X.iloc[test_index]


    print("\nTrain/Test split:")

    print(
        f"Training rows: {len(train_index)}"
    )

    print(
        f"Testing rows: {len(test_index)}"
    )


    train_vehicles = set(
        groups.iloc[train_index]
    )

    test_vehicles = set(
        groups.iloc[test_index]
    )


    print(
        f"Training vehicles: "
        f"{len(train_vehicles)}"
    )

    print(
        f"Testing vehicles: "
        f"{len(test_vehicles)}"
    )

    print(
        "Vehicles in both sets:",
        len(
            train_vehicles.intersection(
                test_vehicles
            )
        )
    )


    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    results = []


    # =====================================================
    # Test each future prediction horizon
    # =====================================================

    for horizon_name, horizon in HORIZONS.items():

        print("\n==================================================")

        print(
            f"Prediction horizon: {horizon_name}"
        )

        print("==================================================")


        # -------------------------------------------------
        # Create future target
        # -------------------------------------------------

        y = create_future_target(
            model_data,
            horizon
        )


        y_train = y.iloc[train_index]

        y_test = y.iloc[test_index]


        print(
            f"Training future derates: "
            f"{int(y_train.sum())}"
        )

        print(
            f"Testing future derates: "
            f"{int(y_test.sum())}"
        )


        print(
            f"Test positive rate: "
            f"{y_test.mean() * 100:.3f}%"
        )


        # -------------------------------------------------
        # Make sure both classes exist
        # -------------------------------------------------

        if (
            y_train.nunique() < 2
            or y_test.nunique() < 2
        ):

            print(
                "ROC AUC cannot be calculated because "
                "both classes are not present."
            )

            continue


        # -------------------------------------------------
        # Random Forest
        # -------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )


        model.fit(
            X_train,
            y_train
        )


        # IMPORTANT:
        # ROC AUC uses predicted probabilities,
        # not the final True/False prediction.
        probabilities = (
            model.predict_proba(X_test)[:, 1]
        )


        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )


        print(
            f"ROC AUC: {roc_auc:.4f}"
        )


        results.append(
            {
                "Prediction_Horizon": horizon_name,
                "Future_Derate_Train_Rows": int(
                    y_train.sum()
                ),
                "Future_Derate_Test_Rows": int(
                    y_test.sum()
                ),
                "Test_Positive_Rate": (
                    y_test.mean()
                ),
                "ROC_AUC": roc_auc,
            }
        )


    # =====================================================
    # Results
    # =====================================================

    results_df = pd.DataFrame(results)


    print("\n==================================================")
    print("ROC AUC SUMMARY")
    print("==================================================")


    print(
        results_df[
            [
                "Prediction_Horizon",
                "Future_Derate_Test_Rows",
                "ROC_AUC",
            ]
        ].to_string(index=False)
    )


    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUT_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results_df.to_csv(
        OUT_RESULTS,
        index=False
    )


    print(
        f"\nSaved results to:\n{OUT_RESULTS}"
    )


if __name__ == "__main__":
    main()