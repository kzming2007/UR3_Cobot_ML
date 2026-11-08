# 13 최종 평가 지표표

## 목적

연구계획서에 명시한 confusion matrix, Accuracy, Macro F1, Precision, Recall, ROC-AUC, PR-AUC를 기존 `12` prediction에서 재집계한다. 새 모델 학습이나 threshold 선택은 수행하지 않는다.

## Window-level seed 평균

| target | model_variant | seed_count | tn_mean | fp_mean | fn_mean | tp_mean | accuracy_mean | macro_f1_mean | positive_precision_mean | positive_recall_mean | positive_f1_mean | roc_auc_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 3270.0 | 234.0 | 288.0 | 875.0 | 0.8882 | 0.8482 | 0.789 | 0.7524 | 0.7702 | 0.9288 | 0.8515 |
| System_Failure | 1d_cnn_19_raw | 3 | 2977.6667 | 526.3333 | 263.6667 | 899.3333 | 0.8307 | 0.7888 | 0.6307 | 0.7733 | 0.6948 | 0.8827 | 0.7334 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 3456.6667 | 47.3333 | 1147.3333 | 15.6667 | 0.744 | 0.4391 | 0.2482 | 0.0135 | 0.0255 | 0.4202 | 0.2237 |
| ProtectiveStop | rf_19_raw_window | 1 | 3893.0 | 121.0 | 131.0 | 522.0 | 0.946 | 0.8871 | 0.8118 | 0.7994 | 0.8056 | 0.9651 | 0.8706 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 3598.6667 | 415.3333 | 112.0 | 541.0 | 0.887 | 0.8021 | 0.5662 | 0.8285 | 0.6725 | 0.9261 | 0.7877 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 3966.6667 | 47.3333 | 637.3333 | 15.6667 | 0.8533 | 0.4821 | 0.2482 | 0.024 | 0.0437 | 0.4599 | 0.1592 |
| GripLost | rf_19_raw_window | 1 | 4092.0 | 40.0 | 248.0 | 287.0 | 0.9383 | 0.8159 | 0.8777 | 0.5364 | 0.6659 | 0.9267 | 0.7742 |
| GripLost | 1d_cnn_19_raw | 3 | 3708.6667 | 423.3333 | 138.0 | 397.0 | 0.8797 | 0.758 | 0.4854 | 0.7421 | 0.5864 | 0.9089 | 0.6675 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 4069.0 | 63.0 | 535.0 | 0.0 | 0.8719 | 0.4658 | 0.0 | 0.0 | 0.0 | 0.3934 | 0.0866 |

- Random Forest는 고정 seed 1회이고 딥러닝 모델은 3개 seed 평균이다.
- `tn/fp/fn/tp_mean`은 딥러닝에서 seed별 confusion count의 평균이므로 정수형 단일 confusion matrix가 아니다. Seed별 원값은 `13_window_seed_metrics.csv`에 있다.
- ROC-AUC와 PR-AUC는 저장된 window score로 계산했다.

## Event-aware cycle consensus confusion matrix

| target | model_variant | seed_count | tn | fp | fn | tp | accuracy | macro_f1 | event_precision | event_recall | event_f1 | normal_cycle_false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 122 | 5 | 0 | 104 | 0.9784 | 0.9782 | 0.9541 | 1.0 | 0.9765 | 0.0394 |
| System_Failure | 1d_cnn_19_raw | 3 | 104 | 23 | 0 | 104 | 0.9004 | 0.9004 | 0.8189 | 1.0 | 0.9004 | 0.1811 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 114 | 13 | 99 | 5 | 0.5152 | 0.3763 | 0.2778 | 0.0481 | 0.082 | 0.1024 |
| ProtectiveStop | rf_19_raw_window | 1 | 153 | 2 | 2 | 74 | 0.9827 | 0.9804 | 0.9737 | 0.9737 | 0.9737 | 0.0129 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 134 | 21 | 3 | 73 | 0.8961 | 0.8883 | 0.7766 | 0.9605 | 0.8588 | 0.1355 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 142 | 13 | 71 | 5 | 0.6364 | 0.4391 | 0.2778 | 0.0658 | 0.1064 | 0.0839 |
| GripLost | rf_19_raw_window | 1 | 178 | 5 | 3 | 45 | 0.9654 | 0.9482 | 0.9 | 0.9375 | 0.9184 | 0.0273 |
| GripLost | 1d_cnn_19_raw | 3 | 134 | 49 | 0 | 48 | 0.7879 | 0.7537 | 0.4948 | 1.0 | 0.6621 | 0.2678 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 164 | 19 | 48 | 0 | 0.71 | 0.4152 | 0.0 | 0.0 | 0.0 | 0.1038 |

- Event cycle은 실제 positive window를 하나 이상 탐지해야 TP로 계산한다.
- Normal cycle은 어느 window에서든 경보가 발생하면 FP로 계산한다.
- 딥러닝은 3개 seed 중 2개 이상인 consensus, Random Forest는 고정 1회다.
- Cycle score aggregation을 사전 고정하지 않았으므로 cycle ROC-AUC와 PR-AUC는 사후 생성하지 않는다.

## 해석 범위

- Window-level과 cycle-level confusion matrix는 평가 단위가 다르므로 직접 합치지 않는다.
- 이 표는 기존 결과의 보고 지표를 보완하며 새로운 독립 실험 결과가 아니다.
- Primary 결론은 `12_matched_lstm_autoencoder_comparison.md`의 event recall과 정상 cycle 오경보율을 유지한다.
