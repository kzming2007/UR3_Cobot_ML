"""Post-hoc descriptive checks recomputed from saved predictions and the raw data. No model is trained.

Not in the preregistered plans. The same-false-alarm point is read from a curve drawn on test scores, so it is
not a tuned operating point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import CURRENT_COLS, MODEL_FEATURE_COLS, OUTPUT_DIR, SPEED_COLS, TARGET_COLS, clean_for_model, load_model_data, markdown_table

BLOCKS = [1, 2, 3, 4, 5, 7, 8, 9, 10]
KEYS = ["target", "cycle_run", "window_start_step"]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, encoding="utf-8-sig", low_memory=False)


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the 10-step windows of experiment 10 and describe each one."""
    rows = []
    for target in TARGET_COLS:
        clean = clean_for_model(df, MODEL_FEATURE_COLS, target).copy()
        clean["raw_row"] = clean.index
        clean["Timestamp"] = pd.to_datetime(clean["Timestamp"], errors="coerce")
        clean = clean.dropna(subset=["Timestamp"]).sort_values(["cycle_run", "Timestamp"]).reset_index(drop=True)
        for run, g in clean.groupby("cycle_run", sort=True):
            g = g.reset_index(drop=True)
            block = int((int(g["cycle"].iloc[0]) - 1) // 25 + 1)
            if len(g) < 10 or block not in BLOCKS:
                continue
            current, speed = g[CURRENT_COLS].to_numpy(float), g[SPEED_COLS].to_numpy(float)
            y, raw = g[target].astype(int).to_numpy(), g["raw_row"].to_numpy()
            seconds = g["Timestamp"].dt.as_unit("s").astype("int64").to_numpy()
            for s in range(len(g) - 9):
                e = s + 10
                rows.append(
                    {
                        "target": target, "cycle_run": run, "block": block, "window_start_step": s,
                        "window_label": int(y[s:e].max()), "last_step_label": int(y[e - 1]),
                        "spliced": bool(raw[e - 1] - raw[s] != 9), "gap_60s": bool(np.diff(seconds[s:e]).max() >= 60),
                        "current_std": current[s:e].std(0).mean(), "speed_std": speed[s:e].std(0).mean(),
                    }
                )
    return pd.DataFrame(rows)


def cycle_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Median across seeds of each cycle's top score, which equals the 2-of-3 rule at any threshold."""
    per_seed = predictions.groupby(["target", "seed", "cycle_run"]).agg(alert=("score", "max")).reset_index()
    hits = predictions[predictions.label.eq(1)].groupby(["target", "seed", "cycle_run"])["score"].max().rename("detect").reset_index()
    per_seed = per_seed.merge(hits, how="left", on=["target", "seed", "cycle_run"])
    return per_seed.groupby(["target", "cycle_run"]).agg(alert=("alert", "median"), detect=("detect", "median")).reset_index()


def same_false_alarm(rf: pd.DataFrame, cnn: pd.DataFrame, normal: set[int]) -> pd.DataFrame:
    rows, rf_scores, cnn_scores = [], cycle_scores(rf), cycle_scores(cnn)
    for target in TARGET_COLS:
        r, c = rf_scores[rf_scores.target.eq(target)].set_index("cycle_run"), cnn_scores[cnn_scores.target.eq(target)].set_index("cycle_run")
        events = r.index[r.detect.notna()]
        nr, nc = r.loc[sorted(normal)], c.loc[sorted(normal)]
        rf_fa = int((nr.alert > 0.5).sum())
        cut = np.sort(nc.alert.to_numpy())[::-1][rf_fa]
        rf_only, cnn_only = int(((nr.alert > 0.5) & ~(nc.alert > 0.5)).sum()), int((~(nr.alert > 0.5) & (nc.alert > 0.5)).sum())
        labels = np.r_[np.ones(len(events)), np.zeros(len(normal))]
        rows.append(
            {
                "target": target, "fault_cycles": len(events), "rf_detected": int((r.detect > 0.5).sum()), "rf_false_alarms": rf_fa,
                "cnn_detected": int((c.detect > 0.5).sum()), "cnn_false_alarms": int((nc.alert > 0.5).sum()),
                "cnn_cut_at_rf_false_alarms": cut, "cnn_false_alarms_at_cut": int((nc.alert > cut).sum()), "cnn_detected_at_cut": int((c.detect > cut).sum()),
                "rf_cycle_auc": roc_auc_score(labels, np.r_[r.loc[events].alert, nr.alert]), "cnn_cycle_auc": roc_auc_score(labels, np.r_[c.loc[events].alert, nc.alert]),
                "rf_only_alarms": rf_only, "cnn_only_alarms": cnn_only, "mcnemar_exact_p": stats.binomtest(rf_only, rf_only + cnn_only, 0.5).pvalue,
            }
        )
    return pd.DataFrame(rows)


def missing_sensitivity(models: dict[str, pd.DataFrame], windows: pd.DataFrame, normal: set[int], normal_with_missing: set[int]) -> pd.DataFrame:
    rows = []
    for name, predictions in models.items():
        merged = predictions.merge(windows[KEYS + ["spliced"]], on=KEYS, validate="many_to_one")
        full, cut = cycle_scores(merged), cycle_scores(merged[~merged.spliced])
        for target in TARGET_COLS:
            a, b = full[full.target.eq(target)], cut[cut.target.eq(target)]
            kept = a[a.cycle_run.isin(normal - normal_with_missing)]
            no_splice = b[b.cycle_run.isin(normal)]
            rows.append(
                {
                    "model": name, "target": target, "fault_cycles": int(a.detect.notna().sum()),
                    "no_splice_detected": int((b.detect > 0.5).sum()), "no_splice_false_alarms": int((no_splice.alert > 0.5).sum()), "normal_cycles": len(no_splice),
                    "faults_only_in_spliced_windows": int(a.detect.notna().sum() - b.detect.notna().sum()),
                    "false_alarms_without_missing_label_cycles": int((kept.alert > 0.5).sum()), "normal_cycles_without_missing_labels": len(kept),
                }
            )
    return pd.DataFrame(rows)


def ae_movement(torch: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    ae = torch[torch.model.eq("lstm_autoencoder_19_raw") & torch.seed.eq(42) & torch.threshold_quantile.eq(0.95)]
    rows = []
    for target in TARGET_COLS:
        m = ae[ae.target.eq(target)].merge(windows, on=KEYS, validate="one_to_one")
        low = m.speed_std <= m.speed_std.quantile(0.25)
        pos, neg = m.label.eq(1), m.label.eq(0)
        rows.append(
            {
                "target": target, "windows": len(m), "spearman_current_std": stats.spearmanr(m.score, m.current_std)[0], "spearman_speed_std": stats.spearmanr(m.score, m.speed_std)[0],
                "median_score_low_speed": m.score[low].median(), "median_score_rest": m.score[~low].median(),
                "speed_std_positive": m.speed_std[pos].median(), "speed_std_negative": m.speed_std[neg].median(),
                "current_std_positive": m.current_std[pos].median(), "current_std_negative": m.current_std[neg].median(),
                "positive_share_in_low_speed": low[pos].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_model_data()
    df["block"] = (df["cycle"].astype(int) - 1) // 25 + 1
    windows = build_windows(df)
    rf = read("12_matched_rf_window_predictions.csv")
    torch = read("12_matched_torch_window_predictions.csv")
    cnn = torch[torch.model.eq("1d_cnn_19_raw")]

    flags = rf[rf.target.eq("System_Failure")].groupby("cycle_run")["label"].max()
    normal = set(flags[flags.eq(0)].index)
    normal_with_missing = normal & set(df[df.ProtectiveStop.isna()].cycle_run)

    base = rf[rf.target.eq("System_Failure")].groupby("cycle_run").agg(block=("scenario_block_25", "first"), system_failure=("label", "max"))
    base["protective_stop"] = rf[rf.target.eq("ProtectiveStop")].groupby("cycle_run")["label"].max()
    base["grip_lost"] = rf[rf.target.eq("GripLost")].groupby("cycle_run")["label"].max()
    blocks = base.groupby("block").agg(
        cycles=("system_failure", "size"), normal=("system_failure", lambda s: int(s.eq(0).sum())),
        system_failure=("system_failure", "sum"), protective_stop=("protective_stop", "sum"), grip_lost=("grip_lost", "sum"),
    ).reset_index()

    raw_blocks = df.groupby("block")["cycle_run"].nunique().rename("raw_cycle_runs").reset_index()
    in_blocks = df[df.block.isin(BLOCKS)]
    data_gaps = pd.DataFrame(
        [
            {
                "target": target, "windows": len(g), "cycles": g.cycle_run.nunique(), "spliced_windows": int(g.spliced.sum()),
                "spliced_positive_windows": int((g.spliced & g.window_label.eq(1)).sum()), "windows_with_60s_gap": int(g.gap_60s.sum()),
                "positive_windows_with_positive_last_step": int((g.window_label.eq(1) & g.last_step_label.eq(1)).sum()), "positive_windows": int(g.window_label.sum()),
                "rows_missing_label_in_blocks": int(in_blocks.ProtectiveStop.isna().sum()), "normal_cycles_with_missing_labels": len(normal_with_missing),
            }
            for target, g in windows.groupby("target", sort=False)
        ]
    )

    comparison, sizes = read("02_row_vs_window_comparison.csv"), read("02_window_dataset_summary.csv")
    row_rate = {t: df[t].dropna().astype(float).mean() for t in TARGET_COLS}
    prevalence = pd.DataFrame(
        [
            {
                "target": x.target, "feature_level": x.feature_level, "window_size": x.window_size, "pr_auc": x.pr_auc,
                "positive_rate": (rate := row_rate[x.target] if x.feature_level == "row_baseline" else sizes[(sizes.target == x.target) & (sizes.window_size == x.window_size)].positive_rate.iloc[0]),
                "pr_auc_over_rate": x.pr_auc / rate,
            }
            for x in comparison.itertuples()
        ]
    )

    splits = read("04_pre_failure_repeated_split_results.csv")
    pre_failure = pd.DataFrame(
        [
            {
                "target": target, "window_size": g.window_size.iloc[0], "horizon": g.prediction_horizon.iloc[0], "splits": len(g),
                "positive_rate": (p := g.positive_count.sum() / g.test_windows.sum()), "all_positive_f1": 2 * p / (1 + p),
                "positive_f1": g.positive_f1.mean(), "runs_with_true_positive": int(g.tp.gt(0).sum()), "pr_auc": g.pr_auc.mean(), "roc_auc": g.roc_auc.mean(),
            }
            for target, g in splits.groupby("target", sort=False)
        ]
    )

    tables = {
        "row_window_prevalence": prevalence, "raw_block_counts": raw_blocks, "block_composition": blocks, "data_gaps": data_gaps,
        "same_false_alarm": same_false_alarm(rf, cnn, normal),
        "missing_sensitivity": missing_sensitivity({"random_forest": rf, "1d_cnn": cnn}, windows, normal, normal_with_missing),
        "ae_movement": ae_movement(torch, windows), "pre_failure_baseline": pre_failure,
    }
    lines = ["# 22 Post-hoc checks from saved results", "", "Post-hoc and descriptive. No model was trained.", ""]
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"22_{name}.csv", index=False, encoding="utf-8-sig")
        lines += [f"## {name}", "", markdown_table(table.round(4)), ""]
    (OUTPUT_DIR / "22_posthoc_saved_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
