"""Figure 1: fault-cycle recall vs true-normal false-alarm rate, from 16_fault_context_alert_summary.csv."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUTPUT_DIR, REPO_ROOT

TARGETS = ["System_Failure", "ProtectiveStop", "GripLost"]
MODELS = [
    ("rf_19_raw_window", "Random Forest", "o", "#12595E"),
    ("1d_cnn_19_raw", "1D CNN", "s", "#B0482A"),
    ("lstm_autoencoder_19_raw_q95", "LSTM Autoencoder q95", "^", "#4E5288"),
]


def main(out_name: str) -> None:
    summary = pd.read_csv(OUTPUT_DIR / "16_fault_context_alert_summary.csv", encoding="utf-8-sig")
    summary = summary[summary["source"].eq("12_matched_models")]
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for candidate, label, marker, color in MODELS:
            row = summary[summary["candidate"].eq(candidate) & summary["target"].eq(target)].iloc[0]
            ax.scatter(100 * row["true_normal_false_alarm_rate"], 100 * row["event_cycle_recall"],
                       marker=marker, s=70, color=color, label=label, zorder=3)
        ax.set_title(target, fontweight="bold")
        ax.set_xlim(-1, 23)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(-5, 105)
        ax.grid(color="#DDE4E4")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("고장 cycle 재현율 (%)")
    fig.supxlabel("완전 정상 cycle 오경보율 (%)", y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = REPO_ROOT / "research" / "figures" / out_name
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "detection_vs_false_alarm.png")
