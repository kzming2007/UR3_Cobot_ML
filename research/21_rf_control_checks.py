"""Post-hoc Random Forest checks on the fixed 10-step windows.

Not in the preregistered plans; seed 42 on window statistics stays the main result.
All models use the same 19 sensors, 4,667 windows, window labels, nine held-out blocks, SMOTE and 300 trees.
- rf_last_step_19: the last time point of each window (19 values). Most positive windows have a negative last step,
  so this input often cannot see the faulty row.
- rf_row_max_19: a row-level model trained on row labels; a window scores the highest of its ten rows. It sees every
  row but has no window features, so it isolates what the window statistics add.
- rf_19_raw_window seeds 43, 44: window statistics with new seeds, for a 2-of-3 consensus like the deep models.
Seed 42 is refitted only to prove this pipeline reproduces the saved predictions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import BASE_SENSOR_COLS, MODEL_FEATURE_COLS, OUTPUT_DIR, clean_for_model, load_model_data, markdown_table

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("matched_rf_21", HERE / "12_matched_rf_baseline.py")
matched_rf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matched_rf
spec.loader.exec_module(matched_rf)

LAST_STEP, ROW_MAX, STATS = "rf_last_step_19", "rf_row_max_19", "rf_19_raw_window"
RUNS = [(LAST_STEP, 42), (ROW_MAX, 42), (STATS, 42), (STATS, 43), (STATS, 44)]
META = ["cycle", "cycle_run", "cycle_occurrence", "scenario_block_25", "window_start_step", "window_end_step"]
WINDOW = 10


def row_view(data: pd.DataFrame, target: str, x: np.ndarray, y: np.ndarray, meta: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """One entry per window step, with its row label; x[w, k] is step start+k of the window's cycle."""
    clean = clean_for_model(data, MODEL_FEATURE_COLS, target).copy()
    clean["Timestamp"] = pd.to_datetime(clean["Timestamp"], errors="coerce")
    clean = clean.dropna(subset=["Timestamp"]).sort_values(["cycle_run", "Timestamp"])
    clean["step"] = clean.groupby("cycle_run").cumcount()
    labels = clean.set_index(["cycle_run", "step"])[target].astype(int)
    steps = pd.DataFrame(
        {
            "window": np.repeat(np.arange(len(x)), WINDOW),
            "cycle_run": np.repeat(meta["cycle_run"], WINDOW),
            "step": np.repeat(meta["window_start_step"], WINDOW) + np.tile(np.arange(WINDOW), len(x)),
            "block": np.repeat(meta["scenario_block_25"], WINDOW),
        }
    )
    steps["row_label"] = labels.reindex(pd.MultiIndex.from_frame(steps[["cycle_run", "step"]])).to_numpy()
    assert steps["row_label"].notna().all()
    assert np.array_equal(steps.groupby("window")["row_label"].max().to_numpy().astype(int), y), "row labels do not rebuild the window labels"
    return steps, x.reshape(-1, len(BASE_SENSOR_COLS))


def score_row_max(window_module, steps: pd.DataFrame, rows_x: np.ndarray, test_block: int, test_windows: np.ndarray) -> np.ndarray:
    first = ~steps.duplicated(["cycle_run", "step"]).to_numpy()
    train = first & steps.block.ne(test_block).to_numpy()
    y_train = steps.row_label[train].astype(int).reset_index(drop=True)
    model = window_module.build_model("rf_smote", y_train)
    model.fit(rows_x[train], y_train)
    in_test = steps.block.eq(test_block).to_numpy()
    scored = steps.loc[in_test, ["window"]].assign(score=model.predict_proba(rows_x[in_test])[:, 1])
    return scored.groupby("window")["score"].max().loc[test_windows].to_numpy()


def fit_all() -> pd.DataFrame:
    manifest = json.loads(matched_rf.MANIFEST_PATH.read_text(encoding="utf-8"))
    matched_rf.validate_manifest(manifest)
    window_module = matched_rf.load_window_module()
    data = load_model_data()
    frames = []
    for entry in manifest["targets"]:
        target = entry["target"]
        with np.load(matched_rf.CACHE_DIR / entry["file"]) as arrays:
            x = arrays["X"][:, :, : len(BASE_SENSOR_COLS)].astype(np.float64)
            y = arrays["y"].astype(np.int64)
            meta = {name: arrays[name].astype(np.int64) for name in META}
        inputs = {LAST_STEP: x[:, -1, :], STATS: matched_rf.summarize_sequences(x)}
        steps, rows_x = row_view(data, target, x, y, meta)
        for model_name, seed in RUNS:
            window_module.RANDOM_STATE = seed  # build_model reads this for both SMOTE and the forest
            for test_block in manifest["blocks"]:
                test = meta["scenario_block_25"] == int(test_block)
                if model_name == ROW_MAX:
                    scores = score_row_max(window_module, steps, rows_x, int(test_block), np.flatnonzero(test))
                else:
                    y_train = pd.Series(y[~test])
                    model = window_module.build_model("rf_smote", y_train)
                    model.fit(inputs[model_name][~test], y_train)
                    scores = model.predict_proba(inputs[model_name][test])[:, 1]
                frame = pd.DataFrame({name: values[test] for name, values in meta.items()})
                frame.insert(0, "target", target)
                frame.insert(1, "model", model_name)
                frame.insert(2, "seed", seed)
                frame["label"] = y[test]
                frame["score"] = scores
                frame["prediction"] = (scores > matched_rf.DECISION_THRESHOLD).astype(np.int64)
                frames.append(frame)
                print(target, model_name, seed, test_block, flush=True)
    return pd.concat(frames, ignore_index=True)


def check_reproduction(predictions: pd.DataFrame) -> None:
    saved = pd.read_csv(OUTPUT_DIR / "12_matched_rf_window_predictions.csv", encoding="utf-8-sig")
    keys = ["target", "cycle_run", "window_start_step"]
    now = predictions[predictions.model.eq(STATS) & predictions.seed.eq(42)]
    both = now.merge(saved, on=keys, suffixes=("", "_saved"), validate="one_to_one")
    assert len(both) == len(saved), "seed 42 windows differ from the saved run"
    assert np.allclose(both.score, both.score_saved), "seed 42 scores differ from the saved run"
    print("seed 42 reproduces 12_matched_rf_window_predictions.csv:", len(both), "windows")


def cycle_counts(cycles: pd.DataFrame, target: str, normal: set[int], any_fault: set[int]) -> dict:
    alert = set(cycles.loc[cycles.cycle_alert, "cycle_run"])
    events = cycles[cycles.true_event]
    other = any_fault - set(events.cycle_run)
    single = target == "System_Failure"
    return {
        "detected": int(events.event_detected.sum()),
        "fault_cycles": len(events),
        "false_alarms": len(alert & normal),
        "normal_cycles": len(normal),
        "cross_alerts": "" if single else len(alert & other),
        "other_fault_cycles": "" if single else len(other),
    }


def per_cycle(g: pd.DataFrame) -> pd.DataFrame:
    g = g.assign(hit=g.prediction & g.label)
    return g.groupby("cycle_run").agg(alert=("prediction", "max"), hit=("hit", "max"), true_event=("label", "max")).astype(bool).reset_index()


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    sf = predictions[predictions.target.eq("System_Failure") & predictions.model.eq(STATS) & predictions.seed.eq(42)]
    flags = sf.groupby("cycle_run")["label"].max()
    normal, any_fault = set(flags[flags.eq(0)].index), set(flags[flags.eq(1)].index)
    rows = []
    for (target, model, seed), g in predictions.groupby(["target", "model", "seed"], sort=False):
        c = per_cycle(g).rename(columns={"alert": "cycle_alert", "hit": "event_detected"})
        rows.append(
            {
                "target": target, "model": model, "aggregation": f"seed {seed}", **cycle_counts(c, target, normal, any_fault),
                "macro_f1": f1_score(g.label, g.prediction, average="macro"),
                "pr_auc": average_precision_score(g.label, g.score), "roc_auc": roc_auc_score(g.label, g.score),
            }
        )
    for target, g in predictions[predictions.model.eq(STATS)].groupby("target", sort=False):
        votes = pd.concat([per_cycle(s) for _, s in g.groupby("seed")]).groupby("cycle_run").agg(alert=("alert", "sum"), hit=("hit", "sum"), true_event=("true_event", "max")).reset_index()
        c = pd.DataFrame({"cycle_run": votes.cycle_run, "cycle_alert": votes.alert.ge(2), "true_event": votes.true_event, "event_detected": votes.hit.ge(2)})
        rows.append({"target": target, "model": STATS, "aggregation": "2 of 3 seeds", **cycle_counts(c, target, normal, any_fault)})
    return pd.DataFrame(rows)


def main() -> None:
    predictions = fit_all()
    check_reproduction(predictions)
    keep = predictions[~(predictions.model.eq(STATS) & predictions.seed.eq(42))]
    keep.to_csv(OUTPUT_DIR / "21_rf_control_window_predictions.csv", index=False, encoding="utf-8-sig")
    summary = summarize(predictions)
    summary.to_csv(OUTPUT_DIR / "21_rf_control_summary.csv", index=False, encoding="utf-8-sig")
    text = "\n".join(
        [
            "# 21 RF post-hoc control checks", "",
            "Post-hoc, not preregistered. Same 4,667 windows, window labels, nine held-out blocks, SMOTE and 300 trees; alert when score > 0.50.",
            "rf_row_max_19 trains on row labels and scores a window by its highest row. Seed 42 on window statistics reproduced the saved predictions of experiment 12.", "",
            markdown_table(summary.round(4)), "",
        ]
    )
    (OUTPUT_DIR / "21_rf_control_checks.md").write_text(text, encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
