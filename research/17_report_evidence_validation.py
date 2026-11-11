"""Check the submission report against saved results: every snippet below is rebuilt from a result file and must appear verbatim in the report."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd

from common import DATASET_PATH, OUTPUT_DIR, REPO_ROOT, ensure_output_dir, markdown_table


REPORT_PATH = REPO_ROOT / "research" / "2026-11-09_research_report_v0.9.md"
CSV_SHA256 = "C789CDA10ACB354A7C1689F617D94A5F39A93FD8CB6C004AD16D36CEA55A74A3"
XLSX_SHA256 = "F0F10917DE9056908A82CEC1AA459DDF9DA2D2DB70D269B128F99241C8796091"
XLSX_PATH = REPO_ROOT / "dataset" / "dataset_02052023.xlsx"
TARGETS = [("System_Failure", "System Failure", "통합 고장"), ("ProtectiveStop", "Protective Stop", "보호 정지"), ("GripLost", "Grip Lost", "파지 상실")]
MAIN_MODELS = [("rf_19_raw_window", "Random Forest"), ("1d_cnn_19_raw", "1D CNN"), ("lstm_autoencoder_19_raw_q95", "LSTM AE q95")]
CLASSICAL = [("random_forest", "Random Forest"), ("logistic_regression", "Logistic Regression"), ("rbf_svm", "RBF SVM")]
ABLATION = [
    ("all_19", "All sensors (baseline)", 19), ("current_family_only", "Current family only", 7), ("joint_current_only", "Joint current only", 6),
    ("tool_current_only", "Tool current only", 1), ("speed_only", "Joint speed only", 6), ("temperature_only", "Joint temperature only", 6),
    ("drop_current_family", "Drop current family", 12), ("drop_speed", "Drop joint speed", 13), ("drop_temperature", "Drop joint temperature", 13),
    ("drop_tool_current", "Drop tool current", 18),
]
CONTROL = [("rf_last_step_19", "Last step"), ("rf_row_max_19", "Row maximum"), ("rf_19_raw_window", "Window statistics")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, encoding="utf-8-sig", low_memory=False)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> str:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * (centre - half):.1f}~{100 * (centre + half):.1f}%"


def lead(name: str, i: int) -> str:
    return f"| {name} |" if i == 0 else "|  |"


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.2f}%)"


def expected_snippets() -> dict[str, str]:
    """Report text that must equal the saved results, keyed by source and check id."""
    s16 = read("16_fault_context_alert_summary.csv")
    row = lambda src, c, t: s16[(s16.source == src) & (s16.candidate == c) & (s16.target == t)].iloc[0]  # noqa: E731
    order = [t for t, _, _ in TARGETS]
    need: dict[str, str] = {}

    for t, en, _ in TARGETS:
        for i, (c, name) in enumerate(MAIN_MODELS):
            r = row("12_matched_models", c, t)
            cross = "n/a" if t == "System_Failure" else pct(r.cross_fault_alert_cycles, r.other_fault_only_cycles)
            need[f"16|table2_{c}_{t}"] = f"{lead(en, i)} {name} | {pct(r.detected_event_cycles, r.event_cycles)} | {pct(r.true_normal_false_alarm_cycles, r.true_normal_cycles)} | {cross} |"
        for i, (c, name) in enumerate(CLASSICAL):
            r = row("15_classical_models", c, t)
            cross = "n/a" if t == "System_Failure" else f"{100 * r.cross_fault_alert_rate:.2f}%"
            need[f"16|table4_{c}_{t}"] = f"{lead(en, i)} {name} | {100 * r.event_cycle_recall:.2f}% | {100 * r.true_normal_false_alarm_rate:.2f}% | {cross} |"

    w13 = read("13_window_model_summary.csv")
    for t, en, _ in TARGETS:
        for i, (c, name) in enumerate(MAIN_MODELS):
            r = w13[(w13.target == t) & (w13.model_variant == c)].iloc[0]
            need[f"13|table3_{c}_{t}"] = f"{lead(en, i)} {name} | {r.macro_f1_mean:.4f} | {r.pr_auc_mean:.4f} | {r.roc_auc_mean:.4f} |"

    c02, prev = read("02_row_vs_window_comparison.csv"), read("22_row_window_prevalence.csv")
    for t, en, ko in TARGETS:
        rr = c02[(c02.target == t) & (c02.feature_level == "row_baseline")].iloc[0]
        wr = c02[(c02.target == t) & (c02.feature_level == "best_window_feature")].iloc[0]
        rate = prev[prev.target == t].set_index("feature_level").positive_rate
        need[f"02|table1_{t}"] = f"| {en} | {rr.macro_f1:.4f} | {int(wr.window_size)}-step | {wr.macro_f1:.4f} | {rr.pr_auc:.4f} | {wr.pr_auc:.4f} | {100 * rate['row_baseline']:.1f} | {100 * rate['best_window_feature']:.1f} |"
        need[f"02|text_pr_auc_{t}"] = f"{ko} {rr.pr_auc:.4f}에서 {wr.pr_auc:.4f}"

    for v, name, n in ABLATION:
        a, b, c = (row("14_sensor_ablation", v, t) for t in order)
        need[f"16|table5_{v}"] = (
            f"| {name} | {n} | {a.detected_event_cycles} | {a.true_normal_false_alarm_cycles} | {b.detected_event_cycles} | {b.true_normal_false_alarm_cycles} | {b.cross_fault_alert_cycles} "
            f"| {c.detected_event_cycles} | {c.true_normal_false_alarm_cycles} | {c.cross_fault_alert_cycles} |"
        )

    control = read("21_rf_control_summary.csv")
    for t, en, _ in TARGETS:
        for i, (m, name) in enumerate(CONTROL):
            r = control[(control.target == t) & (control.model == m) & (control.aggregation == "seed 42")].iloc[0]
            cross = "n/a" if t == "System_Failure" else f"{int(r.cross_alerts)}/{int(r.other_fault_cycles)}"
            need[f"21|table6_{m}_{t}"] = f"{lead(en, i)} {name} | {r.detected}/{r.fault_cycles} | {r.false_alarms}/{r.normal_cycles} | {cross} | {r.macro_f1:.4f} | {r.pr_auc:.4f} |"
    vote = control[control.aggregation == "2 of 3 seeds"].set_index("target")
    need["21|text_consensus_detected"] = "탐지는 " + ", ".join(f"{vote.loc[t, 'detected']}/{vote.loc[t, 'fault_cycles']}" for t in order) + "로 같았다"
    need["21|text_consensus_false_alarms"] = "완전 정상 오경보는 " + ", ".join(f"{vote.loc[t, 'false_alarms']}/127" for t in order)
    need["21|text_consensus_cross"] = f"교차 경보는 보호 정지 {int(vote.loc['ProtectiveStop', 'cross_alerts'])}/28, 파지 상실 {int(vote.loc['GripLost', 'cross_alerts'])}/56"

    gaps = read("22_data_gaps.csv").set_index("target")
    need["22|text_last_step_share"] = "통합 고장 {:.1f}%, 보호 정지 {:.1f}%, 파지 상실 {:.1f}%뿐".format(*(100 * gaps.loc[t, "positive_windows_with_positive_last_step"] / gaps.loc[t, "positive_windows"] for t in order))
    need["22|text_spliced"] = f"{gaps.loc['GripLost', 'windows']:,}개 중 {gaps.loc['GripLost', 'spliced_windows']}개 구간"
    need["22|text_gap"] = f"60초 이상 간격을 포함한 구간도 {gaps.loc['GripLost', 'windows_with_60s_gap']}개"
    need["22|text_missing_rows"] = f"함께 빠진 {gaps.loc['GripLost', 'rows_missing_label_in_blocks']}행"
    missing_total = gaps.loc["GripLost", "rows_missing_label_total"]
    need["22|text_missing_total"] = f"결측 {missing_total}행 중 나머지 {missing_total - gaps.loc['GripLost', 'rows_missing_label_in_blocks']}행"

    same = read("22_same_false_alarm.csv").set_index("target")
    need["22|text_same_fa"] = "1D CNN의 탐지는 " + ", ".join(f"{same.loc[t, 'cnn_detected_at_cut']}/{same.loc[t, 'fault_cycles']}" for t in order) + "로 랜덤 포레스트의 " + ", ".join(f"{same.loc[t, 'rf_detected']}/{same.loc[t, 'fault_cycles']}" for t in order)
    need["22|text_cycle_auc"] = "랜덤 포레스트 " + ", ".join(f"{same.loc[t, 'rf_cycle_auc']:.4f}" for t in order) + ", 1D CNN " + ", ".join(f"{same.loc[t, 'cnn_cycle_auc']:.4f}" for t in order)
    need["22|text_mcnemar"] = "수는 " + ", ".join(f"{same.loc[t, 'rf_only_alarms']}{'와' if same.loc[t, 'rf_only_alarms'] == 2 else '과'} {same.loc[t, 'cnn_only_alarms']}" for t in order)
    assert (same.mcnemar_exact_p < 0.001).all()

    miss = read("22_missing_sensitivity.csv")
    pick = lambda m, t: miss[(miss.model == m) & (miss.target == t)].iloc[0]  # noqa: E731
    need["22|text_without_missing_rf"] = "랜덤 포레스트 " + ", ".join(f"{pick('random_forest', t).false_alarms_without_missing_label_cycles}/112" for t in order)
    need["22|text_without_missing_cnn"] = "1D CNN " + ", ".join(f"{pick('1d_cnn', t).false_alarms_without_missing_label_cycles}/112" for t in order)
    for model, ko in (("random_forest", "랜덤 포레스트는"), ("1d_cnn", "1D CNN은")):
        need[f"22|text_no_splice_{model}"] = ko + " " + ", ".join(
            f"{pick(model, t).no_splice_detected}/{pick(model, t).fault_cycles}{'와' if pick(model, t).fault_cycles == 104 else '과'} {pick(model, t).no_splice_false_alarms}/127" for t in order
        )

    ae = read("22_ae_movement.csv").set_index("target")
    sf = ae.loc["System_Failure"]
    need["22|text_ae_spearman"] = f"Spearman 상관은 전류 {sf.spearman_current_std:.2f}, 속도 {sf.spearman_speed_std:.2f}"

    pre = read("22_pre_failure_baseline.csv").set_index("target")
    need["22|text_pre_f1"] = "통합 고장 {:.4f}, 보호 정지 {:.4f}, 파지 상실 {:.4f}".format(*(pre.loc[t, "positive_f1"] for t in order))
    need["22|text_pre_rates"] = "양성 비율은 {:.2f}%, {:.2f}%, {:.2f}%".format(*(100 * pre.loc[t, "positive_rate"] for t in order))
    need["22|text_pre_baseline"] = "F1은 {:.4f}, {:.4f}, {:.4f}".format(*(pre.loc[t, "all_positive_f1"] for t in order))
    need["22|text_pre_sf_ranks"] = f"PR-AUC {pre.loc['System_Failure', 'pr_auc']:.4f}, ROC-AUC {pre.loc['System_Failure', 'roc_auc']:.4f}"
    need["22|text_pre_ps_runs"] = f"30회 중 {pre.loc['ProtectiveStop', 'runs_with_true_positive']}회"
    p4 = read("04_pre_failure_repeated_split_summary.csv").set_index("target")
    need["04|text_gl_recall"] = f"평균 재현율은 {p4.loc['GripLost', 'positive_recall_mean']:.4f}"
    need["04|text_ps_positive_counts"] = f"회당 {p4.loc['ProtectiveStop', 'test_positive_count_min']}~{p4.loc['ProtectiveStop', 'test_positive_count_max']}개"

    lengths = read("23_window_length_summary.csv")
    cell = lambda t, m, w: lengths[(lengths.target == t) & (lengths.model == m) & (lengths.window_size == w)].iloc[0]  # noqa: E731
    pair = lambda r: f"{r.detected}/{r.fault_cycles}, {r.false_alarms}/{r.normal_cycles}"  # noqa: E731
    for t, en, _ in TARGETS:
        for i, (m, name) in enumerate(MAIN_MODELS[:2]):
            need[f"23|table8_{m}_{t}"] = f"{lead(en, i)} {name} | " + " | ".join(pair(cell(t, m, w)) for w in (5, 10, 20)) + " |"
    span = lambda t, m, col: (lambda v: f"{min(v)}~{max(v)}건")([int(cell(t, m, w)[col]) for w in (5, 10, 20)])  # noqa: E731
    chain = lambda t, m, col: "→".join(str(int(cell(t, m, w)[col])) for w in (5, 10, 20)) + "건"  # noqa: E731
    RF_LONG, CNN_LONG = "rf_19_raw_window", "1d_cnn_19_raw"
    need["23|text_rf_detect_range"] = "탐지가 " + ", ".join(span(t, RF_LONG, "detected") for t in order)
    need["23|text_rf_fa_range"] = "오경보가 " + ", ".join(span(t, RF_LONG, "false_alarms") for t in order)
    need["23|text_cnn_fa_chain"] = "오경보가 " + ", ".join(chain(t, CNN_LONG, "false_alarms") for t in order)
    need["23|text_cnn_detect_chain"] = "탐지도 20스텝에서 " + ", ".join(chain(t, CNN_LONG, "detected") for t in order)
    need["23|text_gl_cross"] = "교차 경보는 5, 10, 20스텝 순으로 랜덤 포레스트 " + ", ".join(str(int(cell("GripLost", RF_LONG, w).cross_alerts)) for w in (5, 10, 20)) + "건, 1D CNN " + ", ".join(str(int(cell("GripLost", CNN_LONG, w).cross_alerts)) for w in (5, 10, 20)) + "건"

    change = lambda a, b, col: [int(cell(t, m, b)[col]) - int(cell(t, m, a)[col]) for t in order for m in (RF_LONG, CNN_LONG)]  # noqa: E731
    det_5_10, fa_5_10, det_10_20 = change(5, 10, "detected"), change(5, 10, "false_alarms"), change(10, 20, "detected")
    need["23|text_5_to_10"] = f"탐지 변화는 {max(map(abs, det_5_10))}건 이내였고 오경보는 " + ("같거나 줄었다" if max(fa_5_10) <= 0 else "늘기도 했다")
    need["23|text_10_to_20"] = ("여섯 경우 모두" if max(det_10_20) < 0 else "일부") + f" 탐지가 {min(map(abs, det_10_20))}~{max(map(abs, det_10_20))}건 줄었다"
    need["23|text_windows_per_cycle"] = "평균 " + ", ".join(f"{r.windows / r.cycles:.1f}" for r in (cell("System_Failure", RF_LONG, w) for w in (5, 10, 20))) + "개"
    cycle_rows = read("23_cycle_rows.csv").iloc[0]
    need["23|text_cycle_rows"] = f"행 수 중앙값은 {cycle_rows.median_rows:.0f}개이고 30행 이상인 사이클은 {100 * cycle_rows.share_30_or_more:.1f}%"

    same23, curve23 = read("23_same_false_alarm.csv"), read("23_false_alarm_curve.csv")
    series = lambda t, col, widths=(5, 10, 20): ", ".join(str(int(same23[(same23.target == t) & (same23.window_size == w)][col].iloc[0])) for w in widths)  # noqa: E731
    need["23|text_same_fa_sf_ps"] = f"통합 고장 {series('System_Failure', 'cnn_detected_at_cut')}건, 보호 정지 {series('ProtectiveStop', 'cnn_detected_at_cut')}건으로 랜덤 포레스트의 {series('System_Failure', 'rf_detected')}건, {series('ProtectiveStop', 'rf_detected')}건보다 적었다"
    need["23|text_same_fa_gl"] = f"5, 10스텝에서 {series('GripLost', 'cnn_detected_at_cut', (5, 10))}건으로 {series('GripLost', 'rf_detected', (5, 10))}건보다 적었지만"
    point = lambda k: curve23[(curve23.window_size == 20) & (curve23.target == "GripLost") & (curve23.false_alarms == k)].iloc[0]  # noqa: E731
    p1, p2, p3 = point(1), point(2), point(3)
    need["23|text_same_fa_gl_20"] = (
        f"오경보 {series('GripLost', 'rf_false_alarms', (20,))}건에 맞추면 1D CNN {p1.cnn_detected}건과 랜덤 포레스트 {p1.rf_detected}건이었다. "
        f"두 모델의 임계값을 함께 움직여 오경보를 2건으로 맞추면 {'둘 다 ' + str(p2.cnn_detected) + '건' if p2.cnn_detected == p2.rf_detected else '서로 달랐고'}, 3건으로 맞추면 {p3.cnn_detected}건과 {p3.rf_detected}건이었다"
    )
    need["23|text_positive_rates"] = "양성 구간 비율은 " + ", ".join(f"{ko} {100 * cell(t, RF_LONG, 5).positive_rate:.1f}%와 {100 * cell(t, RF_LONG, 20).positive_rate:.1f}%" for t, _, ko in TARGETS)

    blocks = read("22_block_composition.csv")
    ids = pd.read_csv(OUTPUT_DIR / "22_block_composition.csv", encoding="utf-8-sig", dtype=str).set_index("block")
    ids_match = ids.absent_ids.dropna().to_dict() == {"3": "74", "8": "187", "10": "244"} and ids.repeated_ids.dropna().to_dict() == {"9": "224 225", "10": "226 227 229 230 231 232 233"}
    need["22|table7_note_ids"] = "Blocks 3 and 8 lack cycle IDs 74 and 187. Blocks 9 and 10 add the second runs of IDs 224–225 and 226–233 (except 228), and block 10 lacks ID 244." if ids_match else "cycle IDs in the Table 7 note no longer match the data"
    raw = read("22_raw_block_counts.csv").set_index("block").raw_cycle_runs
    for b in blocks.itertuples():
        need[f"22|table7_block_{b.block}"] = f"| {b.block} | {b.cycles} | {b.normal} | {b.system_failure} | {b.protective_stop} | {b.grip_lost} |"
    need["22|table7_total"] = f"| Total | {blocks.cycles.sum()} | {blocks.normal.sum()} | {blocks.system_failure.sum()} | {blocks.protective_stop.sum()} | {blocks.grip_lost.sum()} |"
    need["22|text_excluded_blocks"] = f"6번({raw[6]}개)과 11번({raw[11]}개)"

    block_fa = read("22_block_false_alarm.csv")
    max_rate = lambda m: ", ".join(f"{100 * block_fa[(block_fa.model == m) & (block_fa.target == t)].max_rate.iloc[0]:.1f}%" for t in order)  # noqa: E731
    need["22|text_rf_max_block_fa"] = "최댓값은 랜덤 포레스트 " + max_rate("random_forest")
    need["22|text_cnn_max_block_fa"] = "1D CNN " + max_rate("1d_cnn")
    keep_t, only_t, all_t = (row("14_sensor_ablation", v, "ProtectiveStop") for v in ("drop_temperature", "temperature_only", "all_19"))
    need["16|text_temp_detect"] = f"보호 정지 탐지는 {keep_t.detected_event_cycles}/{keep_t.event_cycles}으로 {'같고' if keep_t.detected_event_cycles == all_t.detected_event_cycles else '달랐고'} 온도만으로는 {only_t.detected_event_cycles}/{only_t.event_cycles}"
    need["16|text_temp_fa"] = f"완전 정상 오경보가 {all_t.true_normal_false_alarm_cycles}/{all_t.true_normal_cycles}에서 {keep_t.true_normal_false_alarm_cycles}/{keep_t.true_normal_cycles}로"

    manifest = json.loads((REPO_ROOT / "research" / ".sequence_cache" / "10_sequence_manifest.json").read_text(encoding="utf-8"))["targets"][0]
    need["10|text_cycles"] = f"최종 비교에 들어간 {manifest['cycle_runs']}개"
    need["10|text_windows"] = f"구간은 모두 {manifest['windows']:,}개"
    ev = {t: row("12_matched_models", "rf_19_raw_window", t).event_cycles for t in order}
    need["16|text_fault_cycles"] = f"통합 고장 {ev['System_Failure']}개, 보호 정지 {ev['ProtectiveStop']}개, 파지 상실 {ev['GripLost']}개"
    need["16|text_normal_cycles"] = f"정의하였고 {row('12_matched_models', 'rf_19_raw_window', 'GripLost').true_normal_cycles}개"
    q90 = {t: row("12_matched_models", "lstm_autoencoder_19_raw_q90", t).event_cycle_recall for t in order}
    need["16|text_ae_q90"] = f"통합 고장 {q90['System_Failure']:.4f}, 보호 정지 {q90['ProtectiveStop']:.4f}, 파지 상실 {q90['GripLost']:.0f}"
    runs = read("12_matched_torch_runs.csv")
    runs = runs[runs.model == "lstm_autoencoder_19_raw"]
    need["12|text_calibration_normals"] = f"{runs.calibration_normal_cycles.min()}~{runs.calibration_normal_cycles.max()}개"
    need["16|text_ae_recall_range"] = f"0~{100 * max(row('12_matched_models', 'lstm_autoencoder_19_raw_q95', t).event_cycle_recall for t in order):.2f}%"
    g = lambda c: row("12_matched_models", c, "GripLost")  # noqa: E731
    rf, cnn = g("rf_19_raw_window"), g("1d_cnn_19_raw")
    need["16|text_wilson_recall"] = f"랜덤 포레스트 {wilson(rf.detected_event_cycles, rf.event_cycles)}, 1D CNN {wilson(cnn.detected_event_cycles, cnn.event_cycles)}"
    need["16|text_wilson_fa"] = f"{wilson(rf.true_normal_false_alarm_cycles, 127)}와 {wilson(cnn.true_normal_false_alarm_cycles, 127)}"
    b1 = read("01_baseline_results.csv")
    b1 = b1[b1.model == "rf_smote"].set_index(["target", "split"])["macro_f1"]
    for t, _, ko in TARGETS:
        need[f"01|text_split_{t}"] = f"{ko}{'는' if t == 'ProtectiveStop' else ''} {b1[(t, 'random_stratified')]:.4f}→{b1[(t, 'cycle_group')]:.4f}"
    aer = w13[w13.model_variant == "lstm_autoencoder_19_raw_q95"].roc_auc_mean
    need["13|text_ae_roc"] = f"ROC-AUC는 {aer.min():.2f}~{aer.max():.2f}"
    tool = [row("14_sensor_ablation", "tool_current_only", t).true_normal_false_alarm_rate for t in order]
    need["16|text_tool_only"] = f"{int(100 * min(tool))}~{int(100 * max(tool))}%"

    recall_or_precision = lambda col, m: ", ".join(f"{100 * w13[(w13.target == t) & (w13.model_variant == m)][col].iloc[0]:.1f}%" for t in order)  # noqa: E731
    need["13|text_window_recall"] = f"구간 재현율은 랜덤 포레스트 {recall_or_precision('positive_recall_mean', 'rf_19_raw_window')}, 1D CNN {recall_or_precision('positive_recall_mean', '1d_cnn_19_raw')}이고"
    need["13|text_window_precision"] = f"구간 정밀도는 랜덤 포레스트 {recall_or_precision('positive_precision_mean', 'rf_19_raw_window')}, 1D CNN {recall_or_precision('positive_precision_mean', '1d_cnn_19_raw')}이다"
    audit = read("03_pre_failure_window_summary.csv")
    audit = audit[(audit.target == "ProtectiveStop") & (audit.window_size == 10) & (audit.prediction_horizon == 3)].iloc[0]
    need["03|text_ps_pre_cycles"] = f"보호 정지가 있는 사이클 {audit.cycle_runs_with_event}개 중 최초 양성 이전에 10스텝 구간을 만들 수 있는 사이클은 {audit.cycle_runs_with_pre_event_window}개뿐"

    rec = read("25_recording_time.csv").iloc[0]
    assert rec.dates == 1 and rec.rows_without_time == 0 and rec.nonpositive_gaps == 0 and rec.gap_cycle == rec.gap_previous_cycle
    need["25|text_one_day"] = f"기록은 모두 {rec.date} 하루의 것이고 블록이 바뀌는 지점의 시간 간격은 최대 {rec.largest_block_boundary_gap_seconds:.1f}초"
    need["25|text_gap"] = f"사이클 {rec.gap_cycle} 안의 {rec.largest_gap_hours:.1f}시간 한 번이고 그 밖의 간격은 최대 {rec.second_gap_seconds:.0f}초"
    need["25|text_temperature"] = f"1번 블록 {rec.temperature_first_block:.1f}도에서 10번 블록 {rec.temperature_last_block:.1f}도"
    need["25|text_cycle_id"] = f"사이클 번호의 최댓값은 {rec.largest_cycle_id}이다"

    forward = read("25_forward_split.csv")
    split = lambda name: [forward[(forward.inputs == name) & (forward.target == t)].iloc[0] for t in order]  # noqa: E731
    full, cool = split("all sensors"), split("no temperature")
    counts = lambda rows, a, b: ", ".join(f"{r[a]}/{r[b]}" for r in rows)  # noqa: E731
    assert all(r.false_alarms == 0 for r in full)
    need["25|text_forward"] = f"공백 앞의 사이클 {full[0].train_cycles}개로 학습"
    need["25|text_forward_detect"] = f"0.50 기준의 탐지는 {counts(full, 'detected', 'fault_cycles')}로 크게 줄었고 완전 정상 오경보는 0/{full[0].normal_cycles}이었다"
    need["25|text_forward_abstract"] = f"it detected {full[0].detected} of {full[0].fault_cycles}, {full[1].detected} of {full[1].fault_cycles}, and {full[2].detected} of {full[2].fault_cycles} fault cycles (System Failure, Protective Stop, Grip Lost) at its 0.50 threshold, with no false alarms on {full[0].normal_cycles} fully normal cycles"
    need["25|text_forward_auc"] = "사이클 AUC는 " + ", ".join(f"{r.cycle_auc:.4f}" for r in full)
    need["25|text_forward_no_temperature"] = f"탐지는 {counts(cool, 'detected', 'fault_cycles')}, 완전 정상 오경보는 {counts(cool, 'false_alarms', 'normal_cycles')}이었다"
    need["25|text_forward_train"] = f"보호 정지 {full[1].train_fault_cycles}개, 파지 상실 {full[2].train_fault_cycles}개"

    length = read("25_cycle_length.csv").set_index("target").astype(object)
    ps_len, sf_len, gl_len = (length.loc[t] for t in ("ProtectiveStop", "System_Failure", "GripLost"))
    assert ps_len.train_cut_detected == ps_len.detected and ps_len.train_cut_false_alarms == ps_len.false_alarms and ps_len.rf_alerts_before_fault == 0
    need["25|text_length_rows"] = f"행 수 중앙값은 {ps_len.median_rows_fault:.1f}개로 완전 정상의 {ps_len.median_rows_normal:.0f}개"
    need["25|text_length_rule"] = f"{ps_len.cut}행 이상을 경보로 하면 탐지 {ps_len.detected}/{ps_len.fault_cycles}, 완전 정상 오경보 {ps_len.false_alarms}/{ps_len.normal_cycles}, 교차 경보 {ps_len.cross_alerts}/{ps_len.other_fault_cycles}이었다(사이클 AUC {ps_len.cycle_auc:.4f})"
    need["25|text_length_others"] = f"통합 고장은 {sf_len.detected}/{sf_len.fault_cycles}, 파지 상실은 {gl_len.detected}/{gl_len.fault_cycles}"
    need["25|text_length_delay"] = f"중앙값 {ps_len.rf_median_alert_rows_after_fault:.0f}행 뒤였고 {ps_len.rf_alert_cycles}개 사이클"

    folds = read("25_same_folds.csv")
    fold_chain = lambda level: ", ".join(f"{ko} {folds[(folds.target == t) & (folds.level == level)].random_macro_f1.iloc[0]:.4f}→{folds[(folds.target == t) & (folds.level == level)].cycle_macro_f1.iloc[0]:.4f}" for t, _, ko in TARGETS)  # noqa: E731
    assert (folds.cycle_macro_f1 < folds.random_macro_f1).all()
    need["25|text_same_folds"] = f"행 입력의 {fold_chain('rows')}, 구간 통계의 {fold_chain('windows')}로 모두 낮아졌다"

    fit, by_epochs = read("25_autoencoder_fit.csv"), read("25_autoencoder.csv")
    capped, longer = (by_epochs[by_epochs.epochs == e].set_index("target").loc[order].astype(object) for e in ("at most 40", "400"))
    rf_rows = [row("12_matched_models", "rf_19_raw_window", t) for t in order]
    for t, rf_row in zip(order, rf_rows):
        saved = row("12_matched_models", "lstm_autoencoder_19_raw_q95", t)
        assert capped.loc[t, "detected"] == saved.detected_event_cycles and capped.loc[t, "false_alarms"] == saved.true_normal_false_alarm_cycles  # the summary reproduces Table 2
        assert round(capped.loc[t, "window_roc_auc"], 4) == round(w13[(w13.target == t) & (w13.model_variant == "lstm_autoencoder_19_raw_q95")].roc_auc_mean.iloc[0], 4)
        assert longer.loc[t, "detected"] < rf_row.detected_event_cycles and longer.loc[t, "false_alarms"] > rf_row.true_normal_false_alarm_cycles
    assert (fit.saved_loss_min > fit.window_mean_loss).all() and longer.false_alarms.nunique() == 1
    example = fit[fit.test_block == 2].iloc[0]
    need["25|text_ae_cap"] = f"{fit.ae_runs.sum()}회 학습 중 {fit.ae_runs_at_cap.sum()}회가 상한에 닿았고"
    need["25|text_ae_example"] = f"(예: {example.saved_loss_min:.4f}과 {example.window_mean_loss:.4f})"
    need["25|text_cnn_cap"] = f"1D CNN은 {fit.cnn_runs.sum()}회 중 {fit.cnn_runs_at_cap.sum()}회만 상한에 닿아"
    need["25|text_ae_400_auc"] = "구간 ROC-AUC는 " + ", ".join(f"{longer.loc[t, 'window_roc_auc']:.4f}" for t in order)
    ae_detected = ", ".join(f"{longer.loc[t, 'detected']}/{longer.loc[t, 'fault_cycles']}" for t in order)
    need["25|text_ae_400_counts"] = f"탐지는 {ae_detected}, 완전 정상 오경보는 {longer.false_alarms.iloc[0]}/127, 교차 경보는 보호 정지 {longer.loc['ProtectiveStop', 'cross_alerts']}/28, 파지 상실 {longer.loc['GripLost', 'cross_alerts']}/56"
    need["25|text_ae_400_answer"] = f"탐지는 {ae_detected}로 늘었지만"
    assert all(capped.loc[t, "detected"] < longer.loc[t, "detected"] for t in order)
    need["25|text_ae_400_abstract"] = "a post hoc rerun with 400 epochs improved its detection, which stayed below that of the Random Forest"
    speed_gl = row("14_sensor_ablation", "speed_only", "GripLost")
    need["16|text_speed_abstract"] = f"joint speed alone detected {speed_gl.detected_event_cycles} with {speed_gl.true_normal_false_alarm_cycles} false alarms"
    rf_gl, cnn_gl = row("12_matched_models", "rf_19_raw_window", "GripLost"), row("12_matched_models", "1d_cnn_19_raw", "GripLost")
    drop_gl = row("14_sensor_ablation", "drop_current_family", "GripLost")
    need["16|text_gl_abstract"] = f"the Random Forest and 1D CNN detected {rf_gl.detected_event_cycles} and {cnn_gl.detected_event_cycles} of {rf_gl.event_cycles} fault cycles, with false alarms on {rf_gl.true_normal_false_alarm_cycles} and {cnn_gl.true_normal_false_alarm_cycles} of {rf_gl.true_normal_cycles} fully normal cycles"
    need["16|text_drop_abstract"] = f"Grip Lost from {rf_gl.detected_event_cycles} to {drop_gl.detected_event_cycles} of {rf_gl.event_cycles}"
    length = read("25_cycle_length.csv")
    ps_len = length[length.target == "ProtectiveStop"].iloc[0]
    need["25|text_length_conclusion"] = f"보호 정지는 사이클의 행 수만으로도 {ps_len.detected}/{ps_len.fault_cycles}을 탐지할 수 있어"
    need["03|text_ps_all_blocks"] = f"제외한 6번, 11번 블록까지 포함한 전체 자료에서 보호 정지가 있는 사이클 {audit.cycle_runs_with_event}개"
    gl_rf, gl_cnn = row("12_matched_models", "rf_19_raw_window", "GripLost"), row("12_matched_models", "1d_cnn_19_raw", "GripLost")
    negatives = gl_rf.true_normal_cycles + gl_rf.other_fault_only_cycles
    need["16|text_prereg_false_alarm"] = (
        f"파지 상실에서 랜덤 포레스트 {gl_rf.true_normal_false_alarm_cycles + gl_rf.cross_fault_alert_cycles}/{negatives}, "
        f"1D CNN {gl_cnn.true_normal_false_alarm_cycles + gl_cnn.cross_fault_alert_cycles}/{negatives}"
    )
    runs = read("12_matched_torch_runs.csv")
    early = runs[runs.model.str.contains("auto", case=False) & (runs.best_epoch < 40)].best_epoch
    need["12|text_ae_early_stop"] = f"나머지 {len(early)}회는 검증 손실이 {early.min()}~{early.max()}에폭에서 가장 낮았고"
    need["16|text_rf_for_ae"] = "랜덤 포레스트(" + ", ".join(f"{r.detected_event_cycles}/{r.event_cycles}" for r in rf_rows) + ", 오경보 " + ", ".join(f"{r.true_normal_false_alarm_cycles}/127" for r in rf_rows) + ")"
    return need


def main() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    data = pd.read_csv(DATASET_PATH, encoding="utf-8")
    rows = [
        {"check_id": "dataset_csv_sha256", "source": "dataset", "expected": CSV_SHA256, "passed": sha256(DATASET_PATH) == CSV_SHA256},
        {"check_id": "dataset_xlsx_sha256", "source": "dataset", "expected": XLSX_SHA256, "passed": sha256(XLSX_PATH) == XLSX_SHA256},
        {"check_id": "dataset_shape", "source": "dataset", "expected": "7409x24", "passed": data.shape == (7409, 24)},
        {"check_id": "report_csv_hash", "source": "dataset", "expected": CSV_SHA256[:16], "passed": CSV_SHA256[:16] in report},
        {"check_id": "report_xlsx_hash", "source": "dataset", "expected": XLSX_SHA256[:16], "passed": XLSX_SHA256[:16] in report},
    ]
    for key, snippet in expected_snippets().items():
        source, check_id = key.split("|")
        rows.append({"check_id": check_id, "source": source, "expected": snippet, "passed": snippet in report})

    cited: list[int] = []
    for group in re.findall(r"\[(\d+(?:,\d+)*)\]", report.split("## I.")[1].split("## References")[0]):
        cited += [int(n) for n in group.split(",") if int(n) not in cited]
    listed = [int(n) for n in re.findall(r"^\[(\d+)\] ", report, flags=re.M)]
    rows.append({"check_id": "no_preregistration_claim", "source": "report", "expected": "prespecified, not preregistered", "passed": "preregister" not in report.lower()})
    rows.append({"check_id": "references_in_citation_order", "source": "report", "expected": str(listed), "passed": cited == listed})

    frame = pd.DataFrame(rows)
    ensure_output_dir()
    frame.to_csv(OUTPUT_DIR / "17_report_evidence_validation.csv", index=False, encoding="utf-8-sig")
    failures = frame[~frame.passed]
    open_items = re.findall(r"\[[^\]]*(?:기재)\]", report)
    lines = [
        "# 17 보고서 근거 대조", "", f"- 대상: `{REPORT_PATH.name}`", f"- 총 검사: {len(frame)}", f"- 통과: {int(frame.passed.sum())}", f"- 실패: {len(failures)}",
        f"- 제출 전에 채울 자리: {'; '.join(open_items) or '없음'}", "",
        "## 실패 항목", "", markdown_table(failures[["check_id", "source", "expected"]]), "",
        "## 해석 범위", "", "- 저장된 결과와 보고서 문구의 일치만 확인한다. 모델을 다시 학습하지 않으므로 전체 재현 실험을 대신하지 않는다.", "",
    ]
    (OUTPUT_DIR / "17_report_evidence_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"checks={len(frame)} passed={int(frame.passed.sum())} failed={len(failures)} open={open_items}")
    for r in failures.itertuples():
        print("FAILED", r.check_id, "|", r.expected)
    if len(failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
