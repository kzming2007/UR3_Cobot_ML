# 12 동일 센서 입력 기반 LSTM Autoencoder 후속 비교

## 고정 설계

- 사전 고정 문서: `research/2026-08-23_lstm_autoencoder_preregistration.md`.
- 공통 입력: `cycle_run` 경계를 넘지 않는 10-step×19개 원본 센서.
- 모델: SMOTE+Random Forest window summary, supervised 1D CNN, normal-only LSTM Autoencoder.
- Outer 평가: 9개 candidate block을 한 번씩 test로 사용.
- Autoencoder primary threshold: calibration 정상 cycle 최대 reconstruction error의 95th percentile.
- Deep learning cycle consensus: 3개 seed 중 2개 이상.

## Primary consensus 결과

| target | model_variant | seed_count | event_cycle_count | detected_event_cycle_count | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycle_count | false_alarm_cycle_count | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 104 | 104 | 1.0 | 0.9644 | 1.0 | 1.0 | 127 | 5 | 0.0394 | 0.0169 | 0.0889 | 0.1111 |
| ProtectiveStop | rf_19_raw_window | 1 | 76 | 74 | 0.9737 | 0.909 | 0.9928 | 0.5 | 155 | 2 | 0.0129 | 0.0035 | 0.0458 | 0.0952 |
| GripLost | rf_19_raw_window | 1 | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.6667 | 183 | 5 | 0.0273 | 0.0117 | 0.0624 | 0.0952 |
| System_Failure | 1d_cnn_19_raw | 3 | 104 | 104 | 1.0 | 0.9644 | 1.0 | 1.0 | 127 | 23 | 0.1811 | 0.1238 | 0.2571 | 0.7143 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 76 | 73 | 0.9605 | 0.8903 | 0.9865 | 0.5 | 155 | 21 | 0.1355 | 0.0904 | 0.1982 | 0.5 |
| GripLost | 1d_cnn_19_raw | 3 | 48 | 48 | 1.0 | 0.9259 | 1.0 | 1.0 | 183 | 49 | 0.2678 | 0.2089 | 0.3362 | 0.6667 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 104 | 5 | 0.0481 | 0.0207 | 0.1076 | 0.0 | 127 | 13 | 0.1024 | 0.0608 | 0.1673 | 0.2632 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 76 | 5 | 0.0658 | 0.0284 | 0.1449 | 0.0 | 155 | 13 | 0.0839 | 0.0497 | 0.1382 | 0.25 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 48 | 0 | 0.0 | 0.0 | 0.0741 | 0.0 | 183 | 19 | 0.1038 | 0.0675 | 0.1565 | 0.2857 |

## Seed 평균 window 결과

| target | model_variant | seeds | event_cycle_recall_mean | normal_cycle_false_alarm_rate_mean | window_macro_f1_mean | window_positive_f1_mean | window_pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_19_raw_window | 1 | 1.0 | 0.0394 | 0.8482 | 0.7702 | 0.8515 |
| ProtectiveStop | rf_19_raw_window | 1 | 0.9737 | 0.0129 | 0.8871 | 0.8056 | 0.8706 |
| GripLost | rf_19_raw_window | 1 | 0.9375 | 0.0273 | 0.8159 | 0.6659 | 0.7742 |
| System_Failure | 1d_cnn_19_raw | 3 | 0.9968 | 0.2021 | 0.7888 | 0.6948 | 0.7334 |
| ProtectiveStop | 1d_cnn_19_raw | 3 | 0.9649 | 0.1441 | 0.8021 | 0.6725 | 0.7877 |
| GripLost | 1d_cnn_19_raw | 3 | 0.9931 | 0.2714 | 0.758 | 0.5864 | 0.6675 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 0.0449 | 0.1076 | 0.4391 | 0.0255 | 0.2237 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 0.0614 | 0.0882 | 0.4821 | 0.0437 | 0.1592 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 0.0 | 0.1056 | 0.4658 | 0.0 | 0.0866 |

## Autoencoder threshold 민감도

| target | model_variant | seed_count | event_cycle_count | detected_event_cycle_count | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycle_count | false_alarm_cycle_count | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | lstm_autoencoder_19_raw_q90 | 3 | 104 | 6 | 0.0577 | 0.0267 | 0.1202 | 0.0 | 127 | 25 | 0.1969 | 0.137 | 0.2745 | 0.4375 |
| System_Failure | lstm_autoencoder_19_raw_q95 | 3 | 104 | 5 | 0.0481 | 0.0207 | 0.1076 | 0.0 | 127 | 13 | 0.1024 | 0.0608 | 0.1673 | 0.2632 |
| System_Failure | lstm_autoencoder_19_raw_q97.5 | 3 | 104 | 5 | 0.0481 | 0.0207 | 0.1076 | 0.0 | 127 | 12 | 0.0945 | 0.0549 | 0.1579 | 0.2632 |
| ProtectiveStop | lstm_autoencoder_19_raw_q90 | 3 | 76 | 6 | 0.0789 | 0.0367 | 0.1617 | 0.0 | 155 | 25 | 0.1613 | 0.1117 | 0.2273 | 0.4118 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | 3 | 76 | 5 | 0.0658 | 0.0284 | 0.1449 | 0.0 | 155 | 13 | 0.0839 | 0.0497 | 0.1382 | 0.25 |
| ProtectiveStop | lstm_autoencoder_19_raw_q97.5 | 3 | 76 | 5 | 0.0658 | 0.0284 | 0.1449 | 0.0 | 155 | 12 | 0.0774 | 0.0448 | 0.1304 | 0.25 |
| GripLost | lstm_autoencoder_19_raw_q90 | 3 | 48 | 0 | 0.0 | 0.0 | 0.0741 | 0.0 | 183 | 32 | 0.1749 | 0.1267 | 0.2364 | 0.4286 |
| GripLost | lstm_autoencoder_19_raw_q95 | 3 | 48 | 0 | 0.0 | 0.0 | 0.0741 | 0.0 | 183 | 19 | 0.1038 | 0.0675 | 0.1565 | 0.2857 |
| GripLost | lstm_autoencoder_19_raw_q97.5 | 3 | 48 | 0 | 0.0 | 0.0 | 0.0741 | 0.0 | 183 | 18 | 0.0984 | 0.0631 | 0.1501 | 0.2857 |

## Random Forest와 cycle 오류 겹침

| target | comparison_model | error_type | eligible_cycles | rf_error_count | comparison_error_count | shared_error_count | comparison_only_error_count | rf_only_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 1d_cnn_19_raw | false_alarm | 127 | 5 | 23 | 2 | 21 | 3 |
| System_Failure | 1d_cnn_19_raw | event_miss | 104 | 0 | 0 | 0 | 0 | 0 |
| System_Failure | lstm_autoencoder_19_raw_q95 | false_alarm | 127 | 5 | 13 | 1 | 12 | 4 |
| System_Failure | lstm_autoencoder_19_raw_q95 | event_miss | 104 | 0 | 99 | 0 | 99 | 0 |
| ProtectiveStop | 1d_cnn_19_raw | false_alarm | 155 | 2 | 21 | 0 | 21 | 2 |
| ProtectiveStop | 1d_cnn_19_raw | event_miss | 76 | 2 | 3 | 2 | 1 | 0 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | false_alarm | 155 | 2 | 13 | 0 | 13 | 2 |
| ProtectiveStop | lstm_autoencoder_19_raw_q95 | event_miss | 76 | 2 | 71 | 2 | 69 | 0 |
| GripLost | 1d_cnn_19_raw | false_alarm | 183 | 5 | 49 | 3 | 46 | 2 |
| GripLost | 1d_cnn_19_raw | event_miss | 48 | 3 | 0 | 0 | 0 | 3 |
| GripLost | lstm_autoencoder_19_raw_q95 | false_alarm | 183 | 5 | 19 | 0 | 19 | 5 |
| GripLost | lstm_autoencoder_19_raw_q95 | event_miss | 48 | 3 | 48 | 3 | 45 | 0 |

## 사전 고정 해석 규칙 적용

- `System_Failure` RF 대비 1D CNN: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `System_Failure` RF 대비 LSTM Autoencoder q95: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `ProtectiveStop` RF 대비 1D CNN: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `ProtectiveStop` RF 대비 LSTM Autoencoder q95: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- `GripLost` RF 대비 1D CNN: Recall과 오경보가 엇갈려 단순 우위를 정할 수 없다.
- `GripLost` RF 대비 LSTM Autoencoder q95: Random Forest가 recall과 오경보 기준에서 기술적으로 우세했다.
- 위 우세 판단은 recall과 오경보의 방향만 본 기술적 Pareto 비교이며 통계적 유의성을 뜻하지 않는다.

## 실행 범위

- Random Forest run: 27개.
- 1D CNN run: 81개.
- LSTM Autoencoder run: 27개.
- Autoencoder calibration 정상 cycle 수 범위: 13-38개.
- PyTorch: 2.7.1+cu128, CUDA: 12.8, device: cuda.
- 기록된 전체 학습 시간: 526.9초.

## 해석 제한

- Window 4,667개는 겹치므로 독립 표본 수가 아니다. Cycle-level n은 202개다.
- Wilson interval은 cycle 비율의 불확실성을 나타내지만 block/session 상관을 제거하지 않는다.
- 9개 block은 실제 공정조건 정답이 아니라 수집 구간 proxy다.
- 동일 데이터의 기존 결과를 이미 확인했으므로 독립 외부 검증이 아니라 사전 고정한 내부 후속 비교다.
- Autoencoder sensitivity threshold 중 가장 좋은 값을 primary 결과로 교체하지 않는다.
