"""
Assignment Step 1: read in big_nss.csv and visualize the 5 measurement
columns, comparing trucks currently in Derate vs. not.

Output: assignment/outputs/step1_derate_comparison.png
Run: python assignment/step1_visualize.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "big_nss.csv"
OUT_DIR = REPO_ROOT / "assignment" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


MEASUREMENT_COLS = [
    "EngineOilPressure",
    "EngineOilTemperature",
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]


COLOR_FALSE = "#2a78d6"
COLOR_TRUE = "#eb6834"


def main() -> None:
    # Read big_nss.csv into a dataframe named big_nss
    big_nss = pd.read_csv(
        SRC,
        dtype={"EquipmentID": str},
        low_memory=False
    )

    print(f"Loaded big_nss: {big_nss.shape}")

    # Create five box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, MEASUREMENT_COLS):
        data_false = big_nss.loc[
            ~big_nss["Derate"], col
        ].dropna()

        data_true = big_nss.loc[
            big_nss["Derate"], col
        ].dropna()

        bp = ax.boxplot(
            [data_false, data_true],
            tick_labels=["Not in Derate", "In Derate"],
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops={
                "color": "#0b0b0b",
                "linewidth": 1.5
            },
        )

        for patch, color in zip(
            bp["boxes"],
            [COLOR_FALSE, COLOR_TRUE]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#0b0b0b")

        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused sixth subplot
    axes[-1].axis("off")

    fig.suptitle(
        "Measurement Distributions: "
        "In Derate vs. Not in Derate",
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Save visualization
    out_path = OUT_DIR / "step1_derate_comparison.png"
    fig.savefig(out_path, dpi=150)

    print(f"Wrote {out_path}")

    # Print median values for comparison
    print("\n--- Median by Derate status ---")
    print(
        big_nss.groupby("Derate")[
            MEASUREMENT_COLS
        ].median().T
    )


if __name__ == "__main__":
    main()