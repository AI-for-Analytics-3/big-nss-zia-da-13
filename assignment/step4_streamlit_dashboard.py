from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    REPO_ROOT
    / "assignment"
    / "outputs"
    / "step4_roc_auc_results.csv"
)


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Future Derate Prediction",
    layout="wide"
)

st.title("Future Derate Prediction Dashboard")

st.write(
    "This dashboard shows how well the model predicts a future derate "
    "at different prediction horizons."
)


# ---------------------------------------------------------
# Load ROC AUC results
# ---------------------------------------------------------

results = pd.read_csv(RESULTS_FILE)

st.subheader("ROC AUC Results")

st.dataframe(results, use_container_width=True)


# ---------------------------------------------------------
# Prepare chart data
# ---------------------------------------------------------

chart_data = results[
    ["Prediction_Horizon", "ROC_AUC"]
].copy()

chart_data = chart_data.set_index("Prediction_Horizon")


# ---------------------------------------------------------
# ROC AUC line chart
# ---------------------------------------------------------

st.subheader("ROC AUC by Prediction Horizon")

st.line_chart(chart_data)


# ---------------------------------------------------------
# Best prediction horizon
# ---------------------------------------------------------

best_row = results.loc[
    results["ROC_AUC"].idxmax()
]

best_horizon = best_row["Prediction_Horizon"]
best_auc = best_row["ROC_AUC"]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best Prediction Horizon",
        best_horizon
    )

with col2:
    st.metric(
        "Best ROC AUC",
        f"{best_auc:.3f}"
    )


# ---------------------------------------------------------
# One-day result
# ---------------------------------------------------------

one_day = results[
    results["Prediction_Horizon"] == "1 day"
]

st.subheader("One-Day Prediction")

if not one_day.empty:

    one_day_auc = one_day.iloc[0]["ROC_AUC"]

    st.metric(
        "1-Day ROC AUC",
        f"{one_day_auc:.3f}"
    )

    if one_day_auc >= 0.90:
        interpretation = "Excellent discrimination"
    elif one_day_auc >= 0.80:
        interpretation = "Good discrimination"
    elif one_day_auc >= 0.70:
        interpretation = "Fair discrimination"
    elif one_day_auc >= 0.60:
        interpretation = "Weak discrimination"
    else:
        interpretation = "Close to random discrimination"

    st.write(
        f"The one-day model has a ROC AUC of "
        f"{one_day_auc:.3f}. "
        f"This indicates {interpretation.lower()}."
    )


# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------

st.subheader("How to Interpret ROC AUC")

st.write(
    """
ROC AUC measures how well the model ranks vehicles that will experience
a future derate above vehicles that will not.

A value of 0.50 means the model is about as good as random guessing.

A value closer to 1.00 means the model is better at separating future
derate cases from non-derate cases.

ROC AUC does not tell us the exact number of false alarms or missed
derates at a specific decision threshold.
"""
)