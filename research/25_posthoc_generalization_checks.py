"""Post-hoc checks of how far the block results reach. Added after the results were seen; none replaces a main table.

Steps: recording time, a forward split around the long recording gap (Random Forest only), cycle length as a detector,
random and cycle folds on the same data, the autoencoder at 40 and 400 epochs (run 24_autoencoder_fixed_epochs.py first).
Usage: python -X utf8 25_posthoc_generalization_checks.py [step ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import BASE_SENSOR_COLS, MODEL_FEATURE_COLS, OUTPUT_DIR, TARGET_COLS, clean_for_model, load_model_data, markdown_table

CACHE = Path(__file__).with_name(".sequence_cache")
BLOCKS = [1, 2, 3, 4, 5, 7, 8, 9, 10]
CUT_ROWS = 27  # one fixed cut for every target; the train_cut columns instead choose it outside each test block
AE_CAP = 40


def load_module(script: str):
    spec = importlib.util.spec_from_file_location(f"module_{script[:2]}", Path(__file__).with_name(script))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, encoding="utf-8-sig", low_memory=False)


def load_arrays() -> dict[str, dict[str, np.ndarray]]:
    arrays = {}
    for target in TARGET_COLS:
        with np.load(CACHE / f"10_{target}.npz") as data:
            arrays[target] = {name: data[name] for name in data.files}
    return arrays


def normal_cycles(arrays) -> set[int]:
    flags = pd.Series(arrays["System_Failure"]["y"]).groupby(arrays["System_Failure"]["cycle_run"]).max()
    return set(flags.index[flags.eq(0)])


def raw_gaps(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df.sort_index()["Timestamp"], errors="coerce", utc=True).diff().dt.total_seconds()


def recording_time() -> pd.DataFrame:
    df = load_model_data().sort_index()
    times = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    gaps = raw_gaps(df)
    block = (df["cycle"].astype(int) - 1) // 25 + 1
    temperature = df[[c for c in BASE_SENSOR_COLS if c.startswith("Temperature")]].mean(axis=1).groupby(block).mean()
    in_order = gaps.nlargest(2)
    return pd.DataFrame(
        [
            {
                "dates": times.dt.date.nunique(), "date": str(times.dt.date.iloc[0]), "rows_without_time": int(times.isna().sum()), "nonpositive_gaps": int((gaps <= 0).sum()),
                "largest_gap_hours": in_order.iloc[0] / 3600, "gap_row": int(in_order.index[0]), "gap_cycle": int(df.loc[in_order.index[0], "cycle"]), "gap_previous_cycle": int(df["cycle"].shift(1).loc[in_order.index[0]]),
                "second_gap_seconds": in_order.iloc[1], "largest_block_boundary_gap_seconds": gaps[block.ne(block.shift(1))].max(),
                "temperature_first_block": temperature.loc[BLOCKS[0]], "temperature_last_block": temperature.loc[BLOCKS[-1]], "largest_cycle_id": int(df["cycle"].max()),
            }
        ]
    )


def forward_split() -> pd.DataFrame:
    """Train on cycles that end before the long gap, test on cycles that start after it; the cycle holding the gap is dropped."""
    matched = load_module("12_matched_rf_baseline.py")
    windows_module = matched.load_window_module()
    df, arrays = load_model_data(), load_arrays()
    normal = normal_cycles(arrays)
    gap_row = int(raw_gaps(df).idxmax())
    span = df.reset_index().groupby("cycle_run")["index"].agg(["min", "max"])
    before, after = list(span.index[span["max"] < gap_row]), list(span.index[span["min"] >= gap_row])
    no_temperature = [i for i, c in enumerate(BASE_SENSOR_COLS) if not c.startswith("Temperature")]
    rows = []
    for name, columns in (("all sensors", list(range(19))), ("no temperature", no_temperature)):
        for target in TARGET_COLS:
            a = arrays[target]
            runs, labels = a["cycle_run"], a["y"].astype(int)
            x = matched.summarize_sequences(a["X"][:, :, columns].astype(np.float64))
            train, test = np.isin(runs, before), np.isin(runs, after)
            y_train = pd.Series(labels[train])
            model = windows_module.build_model("rf_smote", y_train)
            model.fit(x[train], y_train)
            scored = pd.DataFrame({"run": runs[test], "label": labels[test], "score": model.predict_proba(x[test])[:, 1]})
            alert = scored.groupby("run").score.max()
            hits = scored[scored.label.eq(1)].groupby("run").score.max()
            normals = alert[alert.index.isin(normal)]
            rows.append(
                {
                    "inputs": name, "target": target, "train_cycles": len(np.unique(runs[train])), "train_fault_cycles": int(pd.Series(labels[train]).groupby(runs[train]).max().sum()),
                    "detected": int((hits > 0.5).sum()), "fault_cycles": len(hits), "false_alarms": int((normals > 0.5).sum()), "normal_cycles": len(normals),
                    "cycle_auc": roc_auc_score(np.r_[np.ones(len(hits)), np.zeros(len(normals))], np.r_[alert[hits.index], normals]),
                }
            )
    return pd.DataFrame(rows)


def cycle_length() -> pd.DataFrame:
    """Detection from the number of rows in a cycle alone, with the cut fixed and with the cut chosen outside each test block."""
    rf = read("12_matched_rf_window_predictions.csv")
    all_windows = rf[rf.target.eq("System_Failure")]
    rows_in_cycle = all_windows.groupby("cycle_run").size() + 9
    block = all_windows.groupby("cycle_run")["scenario_block_25"].first()
    label = {t: rf[rf.target.eq(t)].groupby("cycle_run")["label"].max() for t in TARGET_COLS}
    normal = label["System_Failure"].eq(0)
    df = load_model_data()
    out = []
    for target in TARGET_COLS:
        fault = label[target].eq(1)
        other = label["System_Failure"].eq(1) & ~fault
        hit = rows_in_cycle >= CUT_ROWS
        chosen = pd.Series(False, index=rows_in_cycle.index)
        for b in BLOCKS:
            train = block.ne(b)
            cuts = np.sort(rows_in_cycle[train & (fault | normal)].unique())
            balanced = [((rows_in_cycle[train & fault] >= c).mean() + (rows_in_cycle[train & normal] < c).mean()) / 2 for c in cuts]
            chosen[block.eq(b)] = rows_in_cycle[block.eq(b)] >= cuts[int(np.argmax(balanced))]
        clean = clean_for_model(df, MODEL_FEATURE_COLS, target).copy()
        clean["Timestamp"] = pd.to_datetime(clean["Timestamp"], errors="coerce")
        clean = clean.dropna(subset=["Timestamp"]).sort_values(["cycle_run", "Timestamp"])
        first_fault = {run: int(np.argmax(c[target].astype(int).to_numpy())) for run, c in clean.groupby("cycle_run") if c[target].astype(int).max()}
        alerts = rf[rf.target.eq(target) & rf.prediction.eq(1) & rf.cycle_run.isin(first_fault)].groupby("cycle_run").window_start_step.min()
        delay = np.array([start + 9 - first_fault[run] for run, start in alerts.items()])
        out.append(
            {
                "target": target, "median_rows_fault": rows_in_cycle[fault].median(), "median_rows_normal": rows_in_cycle[normal].median(),
                "cut": CUT_ROWS, "detected": int(hit[fault].sum()), "fault_cycles": int(fault.sum()), "false_alarms": int(hit[normal].sum()), "normal_cycles": int(normal.sum()),
                "cross_alerts": int(hit[other].sum()), "other_fault_cycles": int(other.sum()),
                "cycle_auc": roc_auc_score(np.r_[np.ones(fault.sum()), np.zeros(normal.sum())], np.r_[rows_in_cycle[fault], rows_in_cycle[normal]]),
                "train_cut_detected": int(chosen[fault].sum()), "train_cut_false_alarms": int(chosen[normal].sum()),
                "rf_median_alert_rows_after_fault": float(np.median(delay)), "rf_alert_cycles": len(delay), "rf_alerts_before_fault": int((delay < 0).sum()),
            }
        )
    return pd.DataFrame(out)


def same_folds() -> pd.DataFrame:
    """Five random folds and five cycle folds on the same rows, then on the same windows."""
    matched = load_module("12_matched_rf_baseline.py")
    windows_module = matched.load_window_module()
    df, arrays = load_model_data(), load_arrays()
    rows = []
    for target in TARGET_COLS:
        clean = clean_for_model(df, MODEL_FEATURE_COLS, target)
        a = arrays[target]
        data = {
            "rows": (clean[MODEL_FEATURE_COLS].to_numpy(float), clean[target].astype(int).to_numpy(), clean["cycle_run"].to_numpy()),
            "windows": (matched.summarize_sequences(a["X"][:, :, :19].astype(np.float64)), a["y"].astype(int), a["cycle_run"]),
        }
        for level, (x, y, groups) in data.items():
            scores = {}
            for name, folds in (("random", StratifiedKFold(5, shuffle=True, random_state=42).split(x, y)), ("cycle", GroupKFold(5).split(x, y, groups))):
                predicted = np.zeros(len(y), dtype=int)
                for train, test in folds:
                    y_train = pd.Series(y[train])
                    model = windows_module.build_model("rf_smote", y_train)
                    model.fit(x[train], y_train)
                    predicted[test] = model.predict(x[test])
                scores[name] = f1_score(y, predicted, average="macro")
            rows.append({"target": target, "level": level, "units": len(y), "cycles": len(np.unique(groups)), "random_macro_f1": scores["random"], "cycle_macro_f1": scores["cycle"]})
    return pd.DataFrame(rows)


def two_of_three(scored: pd.DataFrame, target: str, normal: set[int], fault_flags: dict[str, pd.Series]) -> dict:
    """The counting rule of Table 2 for the autoencoder: a cycle counts when at least two seeds agree."""
    alert = scored.groupby(["seed", "cycle_run"]).prediction.max().groupby("cycle_run").sum().ge(2)
    hit = scored[scored.label.eq(1)].groupby(["seed", "cycle_run"]).prediction.max().groupby("cycle_run").sum().ge(2)
    other = fault_flags["System_Failure"].eq(1) & fault_flags[target].eq(0)
    aucs = [roc_auc_score(g.label, g.score) for _, g in scored.groupby("seed")]
    return {
        "detected": int(hit.sum()), "fault_cycles": len(hit), "false_alarms": int(alert[alert.index.isin(normal)].sum()), "normal_cycles": len(normal),
        "cross_alerts": int(alert[other[other].index].sum()) if target != "System_Failure" else 0, "other_fault_cycles": int(other.sum()), "window_roc_auc": float(np.mean(aucs)),
    }


def autoencoder() -> pd.DataFrame:
    arrays = load_arrays()
    normal = normal_cycles(arrays)
    frames = {t: pd.DataFrame({"cycle_run": arrays[t]["cycle_run"].astype(int), "window_start_step": arrays[t]["window_start_step"].astype(int), "label": arrays[t]["y"].astype(int)}) for t in TARGET_COLS}
    flags = {t: f.groupby("cycle_run").label.max() for t, f in frames.items()}
    saved = read("12_matched_torch_window_predictions.csv")
    saved = saved[saved.model.eq("lstm_autoencoder_19_raw") & saved.threshold_quantile.eq(0.95)]
    longer = read("24_ae_fixed_epoch_scores.csv")
    longer["prediction"] = (longer.score > longer.threshold_q95).astype(int)
    rows = []
    for target in TARGET_COLS:
        rows.append({"epochs": f"at most {AE_CAP}", "target": target, **two_of_three(saved[saved.target.eq(target)], target, normal, flags)})
        merged = longer.merge(frames[target], on=["cycle_run", "window_start_step"], validate="many_to_one")
        rows.append({"epochs": "400", "target": target, **two_of_three(merged, target, normal, flags)})
    return pd.DataFrame(rows)


def autoencoder_fit() -> pd.DataFrame:
    """Validation loss of the saved autoencoder runs against replacing every sensor by its own mean within the window."""
    runs = read("12_matched_torch_runs.csv")
    ae, cnn = runs[runs.model.eq("lstm_autoencoder_19_raw")], runs[runs.model.eq("1d_cnn_19_raw")]
    with np.load(CACHE / "10_System_Failure.npz") as data:
        x, blocks, cycle, y = data["X"][:, :, :19].astype(float), data["scenario_block_25"], data["cycle_run"], data["y"]
    flags = pd.Series(y).groupby(cycle).max()
    is_normal = np.isin(cycle, flags.index[flags.eq(0)])
    longer = read("24_ae_fixed_epoch_runs.csv")
    rows = []
    for i, test in enumerate(BLOCKS):
        val, calibration = BLOCKS[(i + 1) % 9], [BLOCKS[(i + 2) % 9], BLOCKS[(i + 3) % 9]]
        core = ~np.isin(blocks, [test, val, *calibration]) & is_normal
        flat = x[core].reshape(-1, 19)
        val_x = (x[(blocks == val) & is_normal] - flat.mean(0)) / flat.std(0)
        saved = ae[ae.test_block.eq(test)]
        rows.append(
            {
                "test_block": test, "validation_block": val, "saved_loss_min": saved.best_validation_loss.min(), "saved_loss_max": saved.best_validation_loss.max(),
                "window_mean_loss": ((val_x - val_x.mean(axis=1, keepdims=True)) ** 2).mean(), "ae_runs_at_cap": int(saved.best_epoch.eq(AE_CAP).sum()), "ae_runs": len(saved),
                "cnn_runs_at_cap": int(cnn[cnn.test_block.eq(test)].best_epoch.eq(AE_CAP).sum()), "cnn_runs": int(cnn.test_block.eq(test).sum()),
                "fixed_400_train_mse_mean": longer[longer.test_block.eq(test)].train_mse.mean(), "fixed_400_calibration_normal_mse_mean": longer[longer.test_block.eq(test)].calibration_normal_mse.mean(),
            }
        )
    return pd.DataFrame(rows)


STEPS = {"recording_time": recording_time, "forward_split": forward_split, "cycle_length": cycle_length, "same_folds": same_folds, "autoencoder": autoencoder, "autoencoder_fit": autoencoder_fit}


def main() -> None:
    for step in sys.argv[1:] or STEPS:
        table = STEPS[step]()
        table.to_csv(OUTPUT_DIR / f"25_{step}.csv", index=False, encoding="utf-8-sig")
        print(step, "\n", table.round(4).to_string(), flush=True)
    lines = ["# 25 Post-hoc checks of how far the block results reach", "", "Added after the results were seen. None replaces a main table.", ""]
    for step in STEPS:
        path = OUTPUT_DIR / f"25_{step}.csv"
        if path.exists():
            lines += [f"## {step}", "", markdown_table(read(path.name).round(4)), ""]
    (OUTPUT_DIR / "25_posthoc_generalization_checks.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
