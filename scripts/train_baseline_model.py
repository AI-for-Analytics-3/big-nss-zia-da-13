"""
Baseline model for predicting an upcoming full derate (DerateWithinHorizon,
6h ahead) from engineered telemetry-history features.

Split strategy: BY VEHICLE, not randomly by row. EquipmentIDs are split
train/test (stratified on whether the vehicle has any positive rows), so
no vehicle appears in both sets. This matters because prior EDA showed
positives cluster heavily on repeat-offender vehicles (PriorIncidentCount
much higher for positive rows) — a row-random split would let the model
"see" a vehicle's other incidents during training and then get credit for
recognizing that same vehicle at test time, which overstates how well it
would generalize to a vehicle it has never observed before.

Model: HistGradientBoostingClassifier — handles NaN natively (no need to
impute the ~50% structurally-missing sensor columns), and class_weight=
'balanced' compensates for the ~0.25% positive rate without manual
resampling.

Metrics reported (chosen for the class imbalance — plain accuracy would
be meaningless at 99.75% negative):
  - PR-AUC (average precision) and ROC-AUC
  - Precision/recall/F1 at the default 0.5 threshold
  - Precision & recall if we only acted on the top 1% highest-risk rows
    (mirrors how a fleet-maintenance tool would actually be used: triage
    a small daily watchlist, not a coin-flip threshold)
  - Permutation feature importance (top 15), on a sampled subset of the
    test set for tractability

Run:
    python scripts/train_baseline_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC = DATA_DIR / "model_ready.parquet"

TARGET = "DerateWithinHorizon"
DROP_COLS = {
    "RecordID", "EventTimeStamp", "EquipmentID", "Derate",
    "SuspectTimestamp", "TimeToNextIncident_hours", TARGET,
}
RANDOM_STATE = 42
TEST_VEHICLE_FRACTION = 0.2
TOP_K_FRACTION = 0.01  # "top 1% riskiest rows" triage-list metric


def main() -> None:
    df = pd.read_parquet(SRC)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    print(f"Using {len(feature_cols)} features:")
    for c in feature_cols:
        print(f"  {c}")

    # Cast bool feature columns to int8 for the model.
    for c in feature_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype("int8")

    # --- Vehicle-level split ---
    veh_has_pos = df.groupby("EquipmentID")[TARGET].any()
    veh_ids = veh_has_pos.index.to_numpy(dtype=object)
    train_veh, test_veh = train_test_split(
        veh_ids, test_size=TEST_VEHICLE_FRACTION,
        stratify=veh_has_pos.values, random_state=RANDOM_STATE,
    )
    train_df = df[df["EquipmentID"].isin(train_veh)]
    test_df = df[df["EquipmentID"].isin(test_veh)]

    print(f"\nTrain: {len(train_df)} rows / {len(train_veh)} vehicles "
          f"({int(train_df[TARGET].sum())} positive)")
    print(f"Test:  {len(test_df)} rows / {len(test_veh)} vehicles "
          f"({int(test_df[TARGET].sum())} positive)")

    X_train, y_train = train_df[feature_cols], train_df[TARGET].astype(int)
    X_test, y_test = test_df[feature_cols], test_df[TARGET].astype(int)

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_iter=300,
        early_stopping=True,
    )
    print("\nFitting HistGradientBoostingClassifier ...")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)

    print(f"\nPR-AUC (average precision): {average_precision_score(y_test, proba):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    print("\n--- Classification report @ threshold 0.5 ---")
    print(classification_report(y_test, pred_default, digits=3, zero_division=0))

    # Top-K triage-list view: if we only flagged the riskiest TOP_K_FRACTION
    # of rows, what precision/recall would that give?
    n_top = max(1, int(len(proba) * TOP_K_FRACTION))
    top_idx = np.argsort(proba)[::-1][:n_top]
    top_precision = y_test.values[top_idx].mean()
    top_recall = y_test.values[top_idx].sum() / max(1, y_test.sum())
    print(f"--- Top {TOP_K_FRACTION * 100:.1f}% riskiest rows (n={n_top}) ---")
    print(f"Precision: {top_precision:.3f}  |  Recall: {top_recall:.3f}")

    # --- Permutation importance on a sampled subset of the test set ---
    sample_n = min(50_000, len(X_test))
    X_test_sample = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
    y_test_sample = y_test.loc[X_test_sample.index]
    print(f"\nComputing permutation importance on a {sample_n}-row test sample ...")
    perm = permutation_importance(
        model, X_test_sample, y_test_sample, n_repeats=5,
        random_state=RANDOM_STATE, scoring="average_precision", n_jobs=-1,
    )
    importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
    print("\n--- Top 15 features by permutation importance (drop in PR-AUC when shuffled) ---")
    print(importances.head(15))


if __name__ == "__main__":
    main()
