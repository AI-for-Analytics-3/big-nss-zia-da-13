"""
Assignment Step 2: clean and fill null values in the 5 measurement columns.

Steps:
1. Drop rows where all 5 measurement columns are null.
2. Drop EquipmentIDs where at least 2 of the 3 Ltd columns are entirely null
   for that vehicle.
3. Predict missing Ltd values from other available Ltd values using
   linear regression.
4. Interpolate remaining Ltd nulls within each EquipmentID.
5. Predict missing EngineOil values using the Ltd columns and, when available,
   the other EngineOil column.

Outputs:
data/big_nss_step2_clean.parquet
data/big_nss_step2_clean.csv

Run:
py -3.13 big-nss-zia-da-13\\assignment\\step2_clean_nulls.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC = REPO_ROOT / "data" / "big_nss.csv"

OUT_PARQUET = REPO_ROOT / "data" / "big_nss_step2_clean.parquet"
OUT_CSV = REPO_ROOT / "data" / "big_nss_step2_clean.csv"


# ---------------------------------------------------------
# Columns used in this assignment
# ---------------------------------------------------------

ENGINE_OIL_COLS = [
    "EngineOilPressure",
    "EngineOilTemperature",
]

LTD_COLS = [
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]

MEASUREMENT_COLS = ENGINE_OIL_COLS + LTD_COLS


# ---------------------------------------------------------
# Helper function for regression
# ---------------------------------------------------------

def fit_and_report(
    X: pd.DataFrame,
    y: pd.Series,
    label: str
) -> LinearRegression:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    test_model = LinearRegression()
    test_model.fit(X_train, y_train)

    predictions = test_model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    print(
        f"{label}: "
        f"R^2 = {r2:.3f}, "
        f"rows used = {len(X)}"
    )

    # Fit final model using all available complete data
    final_model = LinearRegression()
    final_model.fit(X, y)

    return final_model


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------
    # Read data
    # -----------------------------------------------------

    big_nss = pd.read_csv(
        SRC,
        dtype={"EquipmentID": str},
        low_memory=False
    )

    big_nss["EventTimeStamp"] = pd.to_datetime(
        big_nss["EventTimeStamp"],
        errors="coerce"
    )

    starting_rows = len(big_nss)
    starting_vehicles = big_nss["EquipmentID"].nunique()

    print("--------------------------------------------------")
    print("Starting dataset")
    print("--------------------------------------------------")
    print(f"Rows: {starting_rows}")
    print(f"Vehicles: {starting_vehicles}")

    print("\nStarting null counts:")
    print(big_nss[MEASUREMENT_COLS].isna().sum())


    # =====================================================
    # STEP 2.1
    # Drop rows where all 5 measurement columns are null
    # =====================================================

    print("\n--------------------------------------------------")
    print("Step 2.1")
    print("Drop rows where all 5 measurements are null")
    print("--------------------------------------------------")

    all_measurements_null = (
        big_nss[MEASUREMENT_COLS]
        .isna()
        .all(axis=1)
    )

    rows_removed = int(all_measurements_null.sum())

    big_nss = big_nss.loc[
        ~all_measurements_null
    ].copy()

    print(f"Rows removed: {rows_removed}")
    print(f"Rows remaining: {len(big_nss)}")


    # =====================================================
    # STEP 2.2
    # Drop EquipmentIDs missing at least 2 full Ltd columns
    # =====================================================

    print("\n--------------------------------------------------")
    print("Step 2.2")
    print("Drop vehicles missing at least 2 entire Ltd columns")
    print("--------------------------------------------------")

    vehicle_ltd_status = (
        big_nss.groupby("EquipmentID")[LTD_COLS]
        .agg(lambda column: column.isna().all())
    )

    number_missing_ltd_columns = (
        vehicle_ltd_status.sum(axis=1)
    )

    bad_vehicles = (
        number_missing_ltd_columns[
            number_missing_ltd_columns >= 2
        ]
        .index
    )

    print(
        f"Vehicles removed: {len(bad_vehicles)}"
    )

    big_nss = big_nss.loc[
        ~big_nss["EquipmentID"].isin(bad_vehicles)
    ].copy()

    print(
        f"Vehicles remaining: "
        f"{big_nss['EquipmentID'].nunique()}"
    )

    print(
        f"Rows remaining: {len(big_nss)}"
    )


    # Sort data by vehicle and time before interpolation
    big_nss = big_nss.sort_values(
        ["EquipmentID", "EventTimeStamp"]
    ).reset_index(drop=True)


    # =====================================================
    # STEP 2.3
    # Predict missing Ltd values
    # =====================================================

    print("\n--------------------------------------------------")
    print("Step 2.3")
    print("Predict missing Ltd values")
    print("--------------------------------------------------")

    complete_ltd = big_nss.dropna(
        subset=LTD_COLS
    )

    for target in LTD_COLS:

        other_columns = [
            column
            for column in LTD_COLS
            if column != target
        ]

        # ---------------------------------------------
        # Case 1:
        # Target missing, both other Ltd values present
        # ---------------------------------------------

        both_present_mask = (
            big_nss[target].isna()
            & big_nss[other_columns].notna().all(axis=1)
        )

        if both_present_mask.any():

            model = fit_and_report(
                complete_ltd[other_columns],
                complete_ltd[target],
                f"{target} predicted from both other Ltd columns"
            )

            big_nss.loc[
                both_present_mask,
                target
            ] = model.predict(
                big_nss.loc[
                    both_present_mask,
                    other_columns
                ]
            )

        # ---------------------------------------------
        # Case 2:
        # Target missing, only one other Ltd is present
        # ---------------------------------------------

        for known_column in other_columns:

            second_column = [
                column
                for column in other_columns
                if column != known_column
            ][0]

            one_present_mask = (
                big_nss[target].isna()
                & big_nss[known_column].notna()
                & big_nss[second_column].isna()
            )

            if one_present_mask.any():

                model = fit_and_report(
                    complete_ltd[[known_column]],
                    complete_ltd[target],
                    f"{target} predicted from {known_column}"
                )

                big_nss.loc[
                    one_present_mask,
                    target
                ] = model.predict(
                    big_nss.loc[
                        one_present_mask,
                        [known_column]
                    ]
                )

    print("\nLtd nulls after regression:")
    print(big_nss[LTD_COLS].isna().sum())


    # =====================================================
    # STEP 2.4
    # Interpolate remaining Ltd nulls
    # =====================================================

    print("\n--------------------------------------------------")
    print("Step 2.4")
    print("Interpolate remaining Ltd nulls by EquipmentID")
    print("--------------------------------------------------")

    for column in LTD_COLS:

        big_nss[column] = (
            big_nss.groupby("EquipmentID")[column]
            .transform(
                lambda values: values.interpolate(
                    method="linear",
                    limit_area="inside"
                )
            )
        )

    print("\nLtd nulls after interpolation:")
    print(big_nss[LTD_COLS].isna().sum())


    # =====================================================
    # STEP 2.5
    # Predict EngineOil null values
    # =====================================================

    print("\n--------------------------------------------------")
    print("Step 2.5")
    print("Predict missing EngineOil values")
    print("--------------------------------------------------")

    complete_all = big_nss.dropna(
        subset=ENGINE_OIL_COLS + LTD_COLS
    )


    # -----------------------------------------------------
    # Case A:
    # Both EngineOil columns are missing
    # Predict each using the 3 Ltd columns
    # -----------------------------------------------------

    both_oil_missing = (
        big_nss[ENGINE_OIL_COLS]
        .isna()
        .all(axis=1)
        & big_nss[LTD_COLS]
        .notna()
        .all(axis=1)
    )

    for target in ENGINE_OIL_COLS:

        target_mask = (
            both_oil_missing
            & big_nss[target].isna()
        )

        if target_mask.any():

            model = fit_and_report(
                complete_all[LTD_COLS],
                complete_all[target],
                f"{target} predicted from Ltd columns"
            )

            big_nss.loc[
                target_mask,
                target
            ] = model.predict(
                big_nss.loc[
                    target_mask,
                    LTD_COLS
                ]
            )


    # -----------------------------------------------------
    # Case B:
    # Only one EngineOil column is missing
    # Use the other EngineOil column plus Ltd columns
    # -----------------------------------------------------

    for target in ENGINE_OIL_COLS:

        other_engine_column = [
            column
            for column in ENGINE_OIL_COLS
            if column != target
        ][0]

        predictors = (
            [other_engine_column]
            + LTD_COLS
        )

        one_oil_missing = (
            big_nss[target].isna()
            & big_nss[other_engine_column].notna()
            & big_nss[LTD_COLS].notna().all(axis=1)
        )

        if one_oil_missing.any():

            model = fit_and_report(
                complete_all[predictors],
                complete_all[target],
                f"{target} predicted from "
                f"{other_engine_column} and Ltd columns"
            )

            big_nss.loc[
                one_oil_missing,
                target
            ] = model.predict(
                big_nss.loc[
                    one_oil_missing,
                    predictors
                ]
            )


    # =====================================================
    # Final null counts
    # =====================================================

    print("\n--------------------------------------------------")
    print("Final null counts")
    print("--------------------------------------------------")

    print("\nEngineOil columns:")
    print(
        big_nss[ENGINE_OIL_COLS]
        .isna()
        .sum()
    )

    print("\nLtd columns:")
    print(
        big_nss[LTD_COLS]
        .isna()
        .sum()
    )


    # =====================================================
    # Save cleaned dataset
    # =====================================================

    print("\n--------------------------------------------------")
    print("Saving cleaned data")
    print("--------------------------------------------------")

    big_nss.to_parquet(
        OUT_PARQUET,
        index=False
    )

    big_nss.to_csv(
        OUT_CSV,
        index=False
    )

    print(f"Saved parquet file:")
    print(OUT_PARQUET)

    print("\nSaved CSV file:")
    print(OUT_CSV)


    # =====================================================
    # Final dataset summary
    # =====================================================

    print("\n--------------------------------------------------")
    print("Final dataset")
    print("--------------------------------------------------")

    print(f"Starting rows: {starting_rows}")
    print(f"Final rows: {len(big_nss)}")

    print(
        f"Starting vehicles: "
        f"{starting_vehicles}"
    )

    print(
        f"Final vehicles: "
        f"{big_nss['EquipmentID'].nunique()}"
    )


if __name__ == "__main__":
    main()