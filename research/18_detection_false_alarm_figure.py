"""Figure: fault-cycle recall vs true-normal false-alarm rate, from 16_fault_context_alert_summary.csv."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUTPUT_DIR, REPO_ROOT

TARGETS = [("System_Failure", "System Failure"), ("ProtectiveStop", "Protective Stop"), ("GripLost", "Grip Lost")]
MODELS = [
    ("rf_19_raw_window", "Random Forest", "o", "#2a78d6"),
    ("1d_cnn_19_raw", "1D CNN", "s", "#eb6834"),
    ("lstm_autoencoder_19_raw_q95", "LSTM AE q95", "^", "#1baf7a"),
]


def main(out_name: str) -> None:
    summary = pd.read_csv(OUTPUT_DIR / "16_fault_context_alert_summary.csv", encoding="utf-8-sig")
    summary = summary[summary["source"].eq("12_matched_models")]
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.5})
    fig, axes = plt.subplots(1, 3, figsize=(3.47, 1.75), sharey=True)
    for ax, (target, title) in zip(axes, TARGETS):
        for candidate, label, marker, color in MODELS:
            row = summary[summary["candidate"].eq(candidate) & summary["target"].eq(target)].iloc[0]
            ax.scatter(100 * row["true_normal_false_alarm_rate"], 100 * row["event_cycle_recall"], marker=marker,
                       s=24, color=color, edgecolors="white", linewidths=0.5, label=label, zorder=3, clip_on=False)
        ax.set_title(title, fontsize=6.5, fontweight="bold", pad=4)
        ax.set_xlim(-1, 23)
        ax.set_xticks([0, 10, 20])
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 50, 100])
        ax.grid(color="#E3E6E8", linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=2)
    axes[0].set_ylabel("Fault-cycle recall (%)", labelpad=2)
    axes[1].set_xlabel("False alarms on fully normal cycles (%)", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handletextpad=0.1, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 1), pad=0.3, w_pad=0.6)
    out = REPO_ROOT / "research" / "figures" / out_name
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "detection_vs_false_alarm.png")
