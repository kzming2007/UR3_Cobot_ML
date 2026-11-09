"""Post-hoc: the LSTM autoencoder of experiment 12 trained for a fixed 400 epochs instead of at most 40.

Same model, block roles, seeds and q95 rule. Not preregistered and does not replace Table 2. Needs PyTorch; the
scores are summarised by 25_posthoc_generalization_checks.py, which runs in the CPU environment.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

RESEARCH = Path(__file__).parent
EPOCHS = 400
spec = importlib.util.spec_from_file_location("matched_torch_24", RESEARCH / "12_matched_torch_models.py")
matched = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matched
spec.loader.exec_module(matched)


def main() -> None:
    device = matched.choose_device("auto")
    cache = RESEARCH / ".sequence_cache"
    blocks = [int(b) for b in json.loads((cache / "10_sequence_manifest.json").read_text(encoding="utf-8"))["blocks"]]
    with np.load(cache / "10_System_Failure.npz") as data:
        x = data["X"][:, :, : matched.RAW_FEATURE_COUNT].astype(np.float32)
        window_blocks, runs = data["scenario_block_25"].astype(np.int64), data["cycle_run"].astype(np.int64)
        starts, labels = data["window_start_step"].astype(np.int64), data["y"].astype(np.int64)
    normal = matched.complete_normal_cycle_mask(runs, labels)

    outputs = RESEARCH / "outputs"
    with (outputs / "24_ae_fixed_epoch_scores.csv").open("w", newline="", encoding="utf-8") as score_file, (outputs / "24_ae_fixed_epoch_runs.csv").open("w", newline="", encoding="utf-8") as run_file:
        scores_out, runs_out = csv.writer(score_file), csv.writer(run_file)
        scores_out.writerow(["seed", "test_block", "cycle_run", "window_start_step", "score", "threshold_q95"])
        runs_out.writerow(["seed", "test_block", "epochs", "train_mse", "calibration_normal_mse", "threshold_q95", "device"])
        for seed in matched.SEEDS:
            for i, test in enumerate(blocks):
                calibration = [blocks[(i + 2) % len(blocks)], blocks[(i + 3) % len(blocks)]]
                is_test, is_calibration = window_blocks == test, np.isin(window_blocks, calibration) & normal
                is_train = ~np.isin(window_blocks, [test, *calibration]) & normal
                train_x, calibration_x, test_x = matched.standardize(x[is_train], x[is_calibration], x[is_test])
                model = matched.fit_autoencoder(train_x, EPOCHS, seed, device)
                calibration_scores = matched.reconstruction_scores(model, calibration_x, device)
                threshold = float(np.quantile(matched.cycle_max_scores(calibration_scores, runs[is_calibration]), 0.95, method="higher"))
                test_scores = matched.reconstruction_scores(model, test_x, device)
                scores_out.writerows(zip([seed] * len(test_scores), [test] * len(test_scores), runs[is_test], starts[is_test], test_scores, [threshold] * len(test_scores)))
                train_mse = float(matched.reconstruction_scores(model, train_x, device).mean())
                runs_out.writerow([seed, test, EPOCHS, train_mse, float(calibration_scores.mean()), threshold, str(device)])
                score_file.flush()
                run_file.flush()
                print(f"seed={seed} test={test} train_mse={train_mse:.4f} q95={threshold:.4f}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
