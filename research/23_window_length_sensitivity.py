"""Post-hoc sensitivity of the main Random Forest and 1D CNN comparison to the window length (5 and 20 steps).

Not in the preregistered plans; the fixed 10-step results stay the main results. Everything except the window length
follows experiment 12: 19 sensors, nine held-out blocks, window label, score > 0.50, Random Forest seed 42, 1D CNN seeds
42, 43, 44 with a 2-of-3 consensus.

Run in this order. The CNN step needs the PyTorch environment, which has no pandas:
  python -X utf8 research\\23_window_length_sensitivity.py rf
  <pytorch python> -X utf8 research\\23_window_length_sensitivity.py cnn
  python -X utf8 research\\23_window_length_sensitivity.py summary
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / ".sequence_cache"
OUTPUT = HERE / "outputs"
LENGTHS = [5, 20]
TARGETS = ["System_Failure", "ProtectiveStop", "GripLost"]
SEEDS = [42, 43, 44]
SENSORS = 19
FIELDS = ["window_size", "target", "model", "seed", "test_block", "cycle_run", "window_start_step", "label", "score", "prediction"]
RF_PATH = OUTPUT / "23_rf_window_predictions.csv"
CNN_PATH = OUTPUT / "23_cnn_window_predictions.csv"


def load(script: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cached(length: int, target: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / f"window_{length}" / f"{target}.npz") as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def rows_for(length, target, model, seed, block, arrays, test, scores):
    return [
        {
            "window_size": length, "target": target, "model": model, "seed": seed, "test_block": block,
            "cycle_run": int(arrays["cycle_run"][i]), "window_start_step": int(arrays["window_start_step"][i]),
            "label": int(arrays["y"][i]), "score": float(score), "prediction": int(score > 0.5),
        }
        for i, score in zip(np.flatnonzero(test), scores)
    ]


def blocks() -> list[int]:
    return [int(b) for b in json.loads((CACHE / "10_sequence_manifest.json").read_text(encoding="utf-8"))["blocks"]]


def run_rf() -> None:
    import pandas as pd

    from common import load_model_data

    prepare, matched = load("10_prepare_sequence_data.py", "prepare_23"), load("12_matched_rf_baseline.py", "matched_23")
    window_module = matched.load_window_module()
    data = load_model_data()
    test_blocks = blocks()

    prepare.WINDOW_SIZE = 10
    for target in TARGETS:
        rebuilt = prepare.build_sequence_dataset(data, target, test_blocks)
        with np.load(CACHE / f"10_{target}.npz") as saved:
            assert all(np.array_equal(rebuilt[k], saved[k]) for k in saved.files), f"10-step rebuild differs for {target}"
    print("10-step rebuild matches the saved windows", flush=True)

    with RF_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for length in LENGTHS:
            prepare.WINDOW_SIZE = length
            (CACHE / f"window_{length}").mkdir(exist_ok=True)
            for target in TARGETS:
                arrays = prepare.build_sequence_dataset(data, target, test_blocks)
                np.savez_compressed(CACHE / f"window_{length}" / f"{target}.npz", **arrays)
                features = matched.summarize_sequences(arrays["X"][:, :, :SENSORS].astype(np.float64))
                for block in test_blocks:
                    test = arrays["scenario_block_25"] == block
                    y_train = pd.Series(arrays["y"][~test])
                    model = window_module.build_model("rf_smote", y_train)
                    model.fit(features[~test], y_train)
                    writer.writerows(rows_for(length, target, "rf_19_raw_window", 42, block, arrays, test, model.predict_proba(features[test])[:, 1]))
                    print(length, target, "rf", block, flush=True)


def run_cnn() -> None:
    torch_models = load("12_matched_torch_models.py", "torch_23")
    device = torch_models.choose_device("cuda")
    test_blocks = blocks()

    def fit_score(x, y, window_blocks, block, seed):
        validation_block = test_blocks[(test_blocks.index(block) + 1) % len(test_blocks)]
        test, validation = window_blocks == block, window_blocks == validation_block
        core = ~(test | validation)
        core_x, val_x = torch_models.standardize(x[core], x[validation])
        best_epoch, _ = torch_models.select_classifier_epoch(core_x, y[core], val_x, y[validation], seed, device)
        final_x, test_x = torch_models.standardize(x[~test], x[test])
        model = torch_models.fit_classifier(final_x, y[~test], best_epoch, seed, device)
        return test, torch_models.classifier_scores(model, test_x, device), best_epoch

    with CNN_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        # One 10-step fold first, so the summary can confirm this path reproduces experiment 12.
        with np.load(CACHE / "10_System_Failure.npz") as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        x = arrays["X"][:, :, :SENSORS].astype(np.float32)
        test, scores, _ = fit_score(x, arrays["y"].astype(np.int64), arrays["scenario_block_25"], test_blocks[0], 42)
        writer.writerows(rows_for(10, "System_Failure", "1d_cnn_19_raw", 42, test_blocks[0], arrays, test, scores))
        handle.flush()
        for length in LENGTHS:
            for target in TARGETS:
                arrays = cached(length, target)
                x = arrays["X"][:, :, :SENSORS].astype(np.float32)
                y = arrays["y"].astype(np.int64)
                for block in test_blocks:
                    for seed in SEEDS:
                        started = time.perf_counter()
                        test, scores, best_epoch = fit_score(x, y, arrays["scenario_block_25"], block, seed)
                        writer.writerows(rows_for(length, target, "1d_cnn_19_raw", seed, block, arrays, test, scores))
                        handle.flush()
                        print(length, target, block, seed, "epoch", best_epoch, f"{time.perf_counter() - started:.1f}s", flush=True)


def summarize_rows(predictions, length, normal, any_fault):
    """Cycle-level counts for one window length, using a 2-of-3 consensus when a model has three seeds."""
    rows = []
    for (target, model), g in predictions.groupby(["target", "model"], sort=False):
        g = g.assign(hit=g.prediction & g.label)
        per_seed = g.groupby(["seed", "cycle_run"]).agg(alert=("prediction", "max"), hit=("hit", "max"), event=("label", "max")).reset_index()
        need = 1 if per_seed.seed.nunique() == 1 else 2
        votes = per_seed.groupby("cycle_run").agg(alert=("alert", "sum"), hit=("hit", "sum"), event=("event", "max"))
        alert = set(votes.index[votes.alert.ge(need)])
        events = votes[votes.event.eq(1)]
        other = any_fault - set(events.index)
        single = target == "System_Failure"
        rows.append(
            {
                "window_size": length, "target": target, "model": model, "windows": int(len(g) / per_seed.seed.nunique()),
                "positive_rate": g.label.mean(), "cycles": votes.shape[0],
                "detected": int(events.hit.ge(need).sum()), "fault_cycles": len(events),
                "false_alarms": len(alert & normal), "normal_cycles": len(normal),
                "cross_alerts": "" if single else len(alert & other), "other_fault_cycles": "" if single else len(other),
            }
        )
    return rows


def run_summary() -> None:
    import pandas as pd

    from common import OUTPUT_DIR, markdown_table

    rf = pd.read_csv(RF_PATH)
    cnn = pd.read_csv(CNN_PATH)
    check = cnn[cnn.window_size.eq(10)]
    cnn = cnn[cnn.window_size.ne(10)]
    saved_rf = pd.read_csv(OUTPUT_DIR / "12_matched_rf_window_predictions.csv", encoding="utf-8-sig")
    saved_torch = pd.read_csv(OUTPUT_DIR / "12_matched_torch_window_predictions.csv", encoding="utf-8-sig", low_memory=False)
    saved_cnn = saved_torch[saved_torch.model.eq("1d_cnn_19_raw")]

    keys = ["target", "seed", "cycle_run", "window_start_step"]
    both = check.merge(saved_cnn, on=keys, suffixes=("", "_saved"), validate="one_to_one")
    assert len(both) == len(check)
    largest = float((both.score - both.score_saved).abs().max())
    decisions_same = bool(both.prediction.eq(both.prediction_saved).all())
    print("10-step CNN check fold: windows", len(both), "largest score difference", largest, "same decisions", decisions_same)

    rows = []
    for length, predictions in [
        (5, pd.concat([rf[rf.window_size.eq(5)], cnn[cnn.window_size.eq(5)]])),
        (10, pd.concat([saved_rf, saved_cnn])),
        (20, pd.concat([rf[rf.window_size.eq(20)], cnn[cnn.window_size.eq(20)]])),
    ]:
        sf = predictions[predictions.target.eq("System_Failure") & predictions.model.eq("rf_19_raw_window")]
        flags = sf.groupby("cycle_run")["label"].max()
        rows += summarize_rows(predictions[["target", "model", "seed", "cycle_run", "label", "prediction"]], length, set(flags[flags.eq(0)].index), set(flags[flags.eq(1)].index))
    summary = pd.DataFrame(rows).sort_values(["target", "model", "window_size"], key=lambda s: s.map({t: i for i, t in enumerate(TARGETS)}) if s.name == "target" else s, ascending=[True, False, True])
    summary.to_csv(OUTPUT_DIR / "23_window_length_summary.csv", index=False, encoding="utf-8-sig")
    with np.load(CACHE / "window_5" / "System_Failure.npz") as arrays:
        rows = pd.Series(arrays["cycle_run"]).value_counts() + 4  # a cycle of n rows gives n - 4 five-step windows
    pd.DataFrame([{"cycles": len(rows), "median_rows": rows.median(), "share_30_or_more": (rows >= 30).mean()}]).to_csv(OUTPUT_DIR / "23_cycle_rows.csv", index=False, encoding="utf-8-sig")
    text = "\n".join(
        [
            "# 23 Window length sensitivity", "",
            "Post-hoc, not preregistered. Same 19 sensors, nine held-out blocks and rules as experiment 12; only the window length changes. 10-step rows are the saved results of experiment 12.",
            f"Check: one 10-step CNN fold rerun through this path differs from experiment 12 by at most {largest:.2e} in score, same decisions: {decisions_same}.", "",
            markdown_table(summary.round(4)), "",
        ]
    )
    (OUTPUT_DIR / "23_window_length_sensitivity.md").write_text(text, encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    {"rf": run_rf, "cnn": run_cnn, "summary": run_summary}[sys.argv[1]]()
