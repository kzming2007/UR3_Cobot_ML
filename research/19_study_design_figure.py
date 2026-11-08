"""Figure: study design. Block roles follow 12_matched_torch_models.py (val = next block, calibration = next two)."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import REPO_ROOT

BLOCKS = [1, 2, 3, 4, 5, 7, 8, 9, 10]
FILL = {"Test": "#2a78d6", "Train": "#EEF1F3", "Val": "#FBE0D5", "Cal": "#D3EFE4"}


def roles(test: int, n_val: int, n_cal: int) -> list[str]:
    out = ["Train"] * len(BLOCKS)
    out[test] = "Test"
    for k in range(n_val):
        out[(test + 1 + k) % len(BLOCKS)] = "Val"
    for k in range(n_cal):
        out[(test + 1 + n_val + k) % len(BLOCKS)] = "Cal"
    return out


def main(out_name: str) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.2})
    fig, ax = plt.subplots(figsize=(3.47, 2.08))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(3.4, 63.4)

    ax.text(0, 61, "(a)", fontweight="bold", va="center")
    steps = ["Public records\n7,409 rows", "Cycles\n231", "10-step windows\n4,667", "Three models\n19 sensors, 9 blocks"]
    widths = [19, 13, 23, 30]
    x = 0.0
    for i, (text, w) in enumerate(zip(steps, widths)):
        ax.add_patch(FancyBboxPatch((x, 45), w, 12, boxstyle="round,pad=0,rounding_size=1.2", fc="#F6F7F8", ec="#8A949B", lw=0.5))
        ax.text(x + w / 2, 51, text, ha="center", va="center", linespacing=1.25)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, 51), (x + w + 4.6, 51), arrowstyle="-|>", mutation_scale=5, lw=0.6, color="#5C666D"))
        x += w + 5

    ax.text(0, 39.5, "(b)", fontweight="bold", va="center")
    left, cell, height, top = 19.0, 9.0, 6.6, 29.5
    ax.text(left - 1.5, top + height + 1.8, "Block", ha="right", va="center", color="#5C666D")
    for j, block in enumerate(BLOCKS):
        ax.text(left + cell * (j + 0.5), top + height + 1.8, str(block), ha="center", va="center", color="#5C666D")
    rows = [("Random Forest", roles(0, 0, 0)), ("1D CNN", roles(0, 1, 0)), ("LSTM AE", roles(0, 1, 2))]
    for i, (name, row) in enumerate(rows):
        y = top - i * (height + 0.8)
        ax.text(left - 1.5, y + height / 2, name, ha="right", va="center")
        for j, role in enumerate(row):
            ax.add_patch(Rectangle((left + cell * j + 0.25, y), cell - 0.5, height, fc=FILL[role], ec="none"))
            ax.text(left + cell * (j + 0.5), y + height / 2, role, ha="center", va="center", color="white" if role == "Test" else "#20282D")
    note = ("1D CNN: choose the epoch on Val, then refit on all 8 blocks.\n"
            "LSTM AE: normal cycles only; refit on Train + Val; Cal sets the alarm threshold.")
    ax.text(0, 8.2, note, va="center", color="#20282D", linespacing=1.35)
    fig.subplots_adjust(0, 0, 1, 1)
    out = REPO_ROOT / "research" / "figures" / out_name
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "study_design.png")
