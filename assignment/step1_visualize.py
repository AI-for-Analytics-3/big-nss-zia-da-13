"""
Assignment Step 1: read in big_nss.csv and visualize the 5 measurement
columns (EngineOilPressure, EngineOilTemperature, DistanceLtd, FuelLtd,
EngineTimeLtd), comparing trucks currently in Derate vs. not.

Chart choice: a box plot per measurement, split by Derate (small
multiples — one axis per subplot, since the 5 measurements live on very
different scales and should never share a single y-axis). Two categories
per subplot (Derate False/True) use the palette's slot-1/slot-2 colors
(blue/orange), which are validated as colorblind-safe adjacent categorical
hues.

Extreme outliers are hidden from the box-plot whiskers (showfliers=False)
purely for readability — the underlying data going into this chart and
every later step is untouched.

Output: assignment/outputs/step1_derate_comparison.png
Run:    python assignment/step1_visualize.py
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

COLOR_FALSE = "#2a78d6"  # categorical slot 1 (blue) — not in derate
COLOR_TRUE = "#eb6834"   # categorical slot 2 (orange) — in derate


def main() -> None:
    big_nss = pd.read_csv(SRC, dtype={"EquipmentID": str}, low_memory=False)
    print(f"Loaded big_nss: {big_nss.shape}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, MEASUREMENT_COLS):
        data_false = big_nss.loc[~big_nss["Derate"], col].dropna()
        data_true = big_nss.loc[big_nss["Derate"], col].dropna()

        bp = ax.boxplot(
            [data_false, data_true],
            tick_labels=["Not in Derate", "In Derate"],
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops={"color": "#0b0b0b", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], [COLOR_FALSE, COLOR_TRUE]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#0b0b0b")

        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=9)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[-1].axis("off")  # 6th subplot slot unused (5 measurements)

    fig.suptitle(
        "Measurement distributions: In Derate vs. Not in Derate\n"
        "(outliers beyond the whiskers hidden for readability; all rows shown, nulls dropped per-column)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = OUT_DIR / "step1_derate_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # Quick numeric summary alongside the plot
    print("\n--- Median by Derate status ---")
    print(big_nss.groupby("Derate")[MEASUREMENT_COLS].median().T)


if __name__ == "__main__":
    main()
