"""
Assignment Step 3:
Predict whether a vehicle is currently in Derate.

Models:
1. Logistic Regression
2. Random Forest

Target:
Derate

Features:
EngineOilPressure
EngineOilTemperature
DistanceLtd
FuelLtd
EngineTimeLtd
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# File path
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC = REPO_ROOT / "data" / "big_nss_step2_clean.parquet"


# ---------------------------------------------------------
# Variables
# ---------------------------------------------------------

FEATURES = [
    "EngineOilPressure",
    "EngineOilTemperature",
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]

TARGET = "Derate"


def main():

    # -----------------------------------------------------
    # Read cleaned data from Step 2
    # -----------------------------------------------------

    big_nss = pd.read_parquet(SRC)

    print("--------------------------------------------------")
    print("Step 3: Predict Derate")
    print("--------------------------------------------------")

    print(f"Rows loaded: {len(big_nss)}")
    print(f"Vehicles: {big_nss['EquipmentID'].nunique()}")


    # -----------------------------------------------------
    # Remove remaining nulls
    # -----------------------------------------------------

    model_data = big_nss.dropna(
        subset=FEATURES + [TARGET, "EquipmentID"]
    ).copy()

    print(
        f"Rows available for modeling: {len(model_data)}"
    )


    # -----------------------------------------------------
    # Look at target distribution
    # -----------------------------------------------------

    print("\n--------------------------------------------------")
    print("Derate distribution")
    print("--------------------------------------------------")

    print(model_data[TARGET].value_counts())

    print("\nDerate percentages:")
    print(
        model_data[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


    # -----------------------------------------------------
    # Define X, y, and vehicle groups
    # -----------------------------------------------------

    X = model_data[FEATURES]

    y = model_data[TARGET].astype(int)

    groups = model_data["EquipmentID"]


    # -----------------------------------------------------
    # Train/test split by EquipmentID
    #
    # This keeps the same vehicle from appearing in both
    # training and testing data.
    # -----------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


    print("\n--------------------------------------------------")
    print("Train/Test Split")
    print("--------------------------------------------------")

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    train_vehicles = set(
        groups.iloc[train_index]
    )

    test_vehicles = set(
        groups.iloc[test_index]
    )

    print(f"Training vehicles: {len(train_vehicles)}")
    print(f"Testing vehicles: {len(test_vehicles)}")

    print(
        "Vehicles appearing in both sets:",
        len(train_vehicles.intersection(test_vehicles))
    )


    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    print("\n==================================================")
    print("LOGISTIC REGRESSION")
    print("==================================================")


    # StandardScaler puts the five measurements on
    # comparable scales before logistic regression.
    logistic_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])


    # Train model
    logistic_model.fit(
        X_train,
        y_train
    )


    # Predict unseen test data
    logistic_predictions = logistic_model.predict(
        X_test
    )


    # Accuracy
    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions
    )

    print(
        f"\nLogistic Regression Accuracy: "
        f"{logistic_accuracy:.4f}"
    )

    print(
        f"Logistic Regression Accuracy %: "
        f"{logistic_accuracy * 100:.2f}%"
    )


    # Confusion matrix
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            logistic_predictions
        )
    )


    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            logistic_predictions,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Logistic Regression Odds Ratios
    # -----------------------------------------------------

    logistic_classifier = (
        logistic_model.named_steps["model"]
    )

    coefficients = (
        logistic_classifier.coef_[0]
    )

    odds_ratios = np.exp(coefficients)

    odds_table = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefficients,
        "Odds_Ratio": odds_ratios,
    })

    odds_table["Percent_Change_in_Odds"] = (
        (odds_table["Odds_Ratio"] - 1) * 100
    )

    print("\n--------------------------------------------------")
    print("Logistic Regression Odds Ratios")
    print("--------------------------------------------------")

    print(
        odds_table
        .sort_values(
            "Odds_Ratio",
            ascending=False
        )
        .to_string(index=False)
    )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    print("\n==================================================")
    print("RANDOM FOREST")
    print("==================================================")


    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )


    # Train model
    random_forest.fit(
        X_train,
        y_train
    )


    # Predict unseen test data
    forest_predictions = random_forest.predict(
        X_test
    )


    # Accuracy
    forest_accuracy = accuracy_score(
        y_test,
        forest_predictions
    )

    print(
        f"\nRandom Forest Accuracy: "
        f"{forest_accuracy:.4f}"
    )

    print(
        f"Random Forest Accuracy %: "
        f"{forest_accuracy * 100:.2f}%"
    )


    # Confusion matrix
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            forest_predictions
        )
    )


    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            forest_predictions,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Random Forest Feature Importance
    # -----------------------------------------------------

    feature_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": random_forest.feature_importances_,
    })

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n--------------------------------------------------")
    print("Random Forest Feature Importance")
    print("--------------------------------------------------")

    print(
        feature_importance.to_string(
            index=False
        )
    )


    # =====================================================
    # Compare models
    # =====================================================

    print("\n==================================================")
    print("MODEL COMPARISON")
    print("==================================================")

    print(
        f"Logistic Regression Accuracy: "
        f"{logistic_accuracy * 100:.2f}%"
    )

    print(
        f"Random Forest Accuracy: "
        f"{forest_accuracy * 100:.2f}%"
    )

    if forest_accuracy > logistic_accuracy:

        print(
            "\nRandom Forest had the higher "
            "test accuracy."
        )

    elif logistic_accuracy > forest_accuracy:

        print(
            "\nLogistic Regression had the higher "
            "test accuracy."
        )

    else:

        print(
            "\nBoth models had the same "
            "test accuracy."
        )


if __name__ == "__main__":
    main()