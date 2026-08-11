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
    page_title="Future Derate Prediction Dashboard",
    layout="wide"
)

st.title("Future Derate Prediction Dashboard")

st.write(
    "This dashboard evaluates how well the model predicts a future derate "
    "at different prediction horizons."
)


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

results = pd.read_csv(RESULTS_FILE)


# ---------------------------------------------------------
# Clean display values
# ---------------------------------------------------------

results["Test_Positive_Percent"] = (
    results["Test_Positive_Rate"] * 100
)


# ---------------------------------------------------------
# Top summary metrics
# ---------------------------------------------------------

best_row = results.loc[
    results["ROC_AUC"].idxmax()
]

best_horizon = best_row["Prediction_Horizon"]
best_auc = best_row["ROC_AUC"]

one_day_row = results[
    results["Prediction_Horizon"] == "1 day"
]

if not one_day_row.empty:
    one_day_auc = one_day_row.iloc[0]["ROC_AUC"]
else:
    one_day_auc = None


st.subheader("Model Summary")

col1, col2, col3 = st.columns(3)

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

with col3:
    if one_day_auc is not None:
        st.metric(
            "1-Day ROC AUC",
            f"{one_day_auc:.3f}"
        )


# ---------------------------------------------------------
# ROC AUC chart
# ---------------------------------------------------------

st.subheader("ROC AUC by Prediction Horizon")

chart_data = results[
    ["Prediction_Horizon", "ROC_AUC"]
].copy()

chart_data = chart_data.set_index(
    "Prediction_Horizon"
)

st.line_chart(
    chart_data,
    height=400
)


# ---------------------------------------------------------
# Interactive horizon selection
# ---------------------------------------------------------

st.subheader("Explore a Prediction Horizon")

selected_horizon = st.selectbox(
    "Choose a prediction horizon:",
    results["Prediction_Horizon"].tolist()
)

selected_row = results[
    results["Prediction_Horizon"] == selected_horizon
].iloc[0]

selected_auc = selected_row["ROC_AUC"]
selected_positive_rate = selected_row["Test_Positive_Percent"]
selected_positive_rows = selected_row["Future_Derate_Test_Rows"]


# ---------------------------------------------------------
# Selected horizon metrics
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ROC AUC",
        f"{selected_auc:.3f}"
    )

with col2:
    st.metric(
        "Future Derate Rows",
        f"{int(selected_positive_rows):,}"
    )

with col3:
    st.metric(
        "Positive Rate",
        f"{selected_positive_rate:.3f}%"
    )


# ---------------------------------------------------------
# Interpretation function
# ---------------------------------------------------------

def interpret_auc(auc):

    if auc >= 0.90:
        return "Excellent discrimination"

    elif auc >= 0.80:
        return "Good discrimination"

    elif auc >= 0.70:
        return "Fair discrimination"

    elif auc >= 0.60:
        return "Weak discrimination"

    else:
        return "Close to random discrimination"


interpretation = interpret_auc(
    selected_auc
)


# ---------------------------------------------------------
# Selected horizon explanation
# ---------------------------------------------------------

st.write(
    f"For the **{selected_horizon}** prediction horizon, "
    f"the model has a ROC AUC of **{selected_auc:.3f}**."
)

st.write(
    f"This indicates **{interpretation.lower()}**."
)

st.write(
    "ROC AUC measures how well the model ranks observations that will "
    "experience a future derate above observations that will not."
)


# ---------------------------------------------------------
# One-day assignment interpretation
# ---------------------------------------------------------

st.subheader("One-Day Prediction Interpretation")

if one_day_auc is not None:

    one_day_interpretation = interpret_auc(
        one_day_auc
    )

    st.write(
        f"The 1-day model has a ROC AUC of "
        f"**{one_day_auc:.3f}**."
    )

    st.write(
        f"This means the model shows "
        f"**{one_day_interpretation.lower()}** "
        f"when distinguishing observations that will experience "
        f"a derate within one day from those that will not."
    )

    st.write(
        "Another way to interpret ROC AUC is as a ranking measure. "
        "A higher value means the model is more likely to assign a "
        "higher risk score to a vehicle that will soon derate than "
        "to one that will not."
    )


# ---------------------------------------------------------
# Results table
# ---------------------------------------------------------

st.subheader("ROC AUC Results Table")

display_table = results[
    [
        "Prediction_Horizon",
        "Future_Derate_Test_Rows",
        "Test_Positive_Percent",
        "ROC_AUC"
    ]
].copy()

display_table = display_table.rename(
    columns={
        "Prediction_Horizon": "Prediction Horizon",
        "Future_Derate_Test_Rows": "Future Derate Test Rows",
        "Test_Positive_Percent": "Positive Rate (%)",
        "ROC_AUC": "ROC AUC"
    }
)

display_table["Positive Rate (%)"] = (
    display_table["Positive Rate (%)"]
    .round(3)
)

display_table["ROC AUC"] = (
    display_table["ROC AUC"]
    .round(3)
)

st.dataframe(
    display_table,
    use_container_width=True
)


# ---------------------------------------------------------
# Compare horizons
# ---------------------------------------------------------

st.subheader("Best to Worst Prediction Horizons")

ranked_results = results.sort_values(
    "ROC_AUC",
    ascending=False
)[
    [
        "Prediction_Horizon",
        "ROC_AUC"
    ]
]

st.bar_chart(
    ranked_results.set_index(
        "Prediction_Horizon"
    )
)


# ---------------------------------------------------------
# ROC AUC explanation
# ---------------------------------------------------------

st.subheader("What ROC AUC Tells Us")

st.write(
    """
ROC AUC measures the model's ability to separate future derate cases
from non-derate cases across many possible classification thresholds.

A ROC AUC of:

- 0.50 is approximately random
- 0.60 to 0.70 is weak
- 0.70 to 0.80 is fair
- 0.80 to 0.90 is good
- 0.90 to 1.00 is excellent
"""
)


# ---------------------------------------------------------
# Limitations
# ---------------------------------------------------------

st.subheader("What ROC AUC Does Not Tell Us")

st.write(
    """
ROC AUC does not tell us:

- the exact number of derates the model will correctly detect
- the number of false alarms at a specific decision threshold
- what probability threshold should trigger maintenance
- whether the predicted probabilities are well calibrated
- whether the model will work equally well for every vehicle
"""
)


# ---------------------------------------------------------
# Key takeaway
# ---------------------------------------------------------

st.subheader("Key Takeaway")

st.write(
    f"The strongest prediction horizon in this analysis is "
    f"**{best_horizon}**, with a ROC AUC of "
    f"**{best_auc:.3f}**."
)

st.write(
    "The dashboard should be used to compare how predictive performance "
    "changes as we try to forecast further ahead of an actual derate."
)