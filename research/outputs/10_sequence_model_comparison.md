# 10 Sequence Model Comparison

## 고정 설계

- 입력: `cycle_run` 경계를 넘지 않는 10-step × 26-feature sequence.
- 외부 평가: 9개 후보 block을 한 번씩 test로 두는 held-out 평가.
- 내부 검증: outer train block 중 다음 순서의 1개 block으로 epoch를 선택한 뒤, 8개 outer train block 전체로 해당 epoch만큼 재학습.
- 모델: 고정 소형 1D CNN, 단층 LSTM, 비교 기준 `SMOTE + Random Forest` window summary.
- Deep learning: class-weighted BCE, Adam, threshold `score > 0.50`, seeds 42/43/44.
- 전체 sensor를 주 설정으로 사용하며 test block은 scaling, epoch 선택, threshold 선택에 사용하지 않는다.
- Window 지표는 seed별로 9개 held-out prediction을 합쳐 계산한 뒤 seed 평균을 낸다. Random Forest는 고정 1회 결과다.

## 결과 요약

| target | model | seeds | event_cycle_recall_mean | event_cycle_recall_min_block_mean | normal_cycle_false_alarm_rate_mean | normal_cycle_false_alarm_rate_max_block_mean | window_macro_f1_mean | window_positive_f1_mean | window_pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 1d_cnn | 3 | 0.9872 | 0.9524 | 0.1575 | 0.5635 | 0.8257 | 0.7417 | 0.7762 |
| System_Failure | lstm | 3 | 0.984 | 0.9109 | 0.1732 | 0.6667 | 0.7939 | 0.6988 | 0.742 |
| ProtectiveStop | 1d_cnn | 3 | 0.943 | 0.6852 | 0.1333 | 0.5417 | 0.8148 | 0.692 | 0.8024 |
| ProtectiveStop | lstm | 3 | 0.9254 | 0.3889 | 0.2172 | 0.631 | 0.8022 | 0.6712 | 0.7103 |
| GripLost | 1d_cnn | 3 | 0.9236 | 0.5556 | 0.2823 | 0.6978 | 0.7496 | 0.5706 | 0.6582 |
| GripLost | lstm | 3 | 0.9514 | 0.7037 | 0.3515 | 0.8444 | 0.7443 | 0.5608 | 0.6484 |
| System_Failure | rf_window_features | 1 | 1.0 | 1.0 | 0.0472 | 0.1111 | 0.8449 | 0.7654 | 0.8513 |
| ProtectiveStop | rf_window_features | 1 | 0.9737 | 0.5 | 0.0129 | 0.0476 | 0.8906 | 0.8117 | 0.8644 |
| GripLost | rf_window_features | 1 | 0.9583 | 0.6667 | 0.0273 | 0.0667 | 0.8275 | 0.6868 | 0.7857 |

## 실행 범위

- Deep learning 학습 run: 162개.
- PyTorch: 2.7.1+cu128, CUDA: 12.8, device: cuda.
- 총 학습 시간: 511.2초.

## 결론

- Random Forest는 세 target 모두에서 두 deep learning 모델보다 정상 cycle 오경보율이 낮고, pooled window positive F1과 PR-AUC가 높았다.
- 1D CNN과 LSTM도 높은 event cycle recall을 보였지만, 정상 cycle 오경보가 누적되어 Random Forest를 대체할 근거는 확인되지 않았다.
- 현재 규모의 공개 데이터에서는 짧은 구간의 통계적 요약과 tree ensemble이 효과적인 기준선이라는 결과로 해석한다.
- 후속 확장은 모델 규모 확대보다 오류가 집중된 held-out block과 false alarm 구간을 먼저 분석하는 것이 타당하다.

## 해석 제한

- Random Forest는 10-step을 통계량으로 정제한 입력이고, 1D CNN/LSTM은 동일 구간의 step 순서를 직접 입력받는다.
- 이 실험은 이상이 이미 포함된 window의 구간 단위 탐지 비교이며 pre-failure 예측이 아니다.
- 모델 구조와 학습 설정은 사전 고정했으며 test 결과를 이용한 architecture 또는 threshold 선택은 수행하지 않는다.
- 후보 block은 실제 공정조건 정답표가 아니라 cycle 번호 기반 proxy다.
- `09` 보고서의 window 지표는 block별 평균이고 이 표는 held-out prediction pooled 지표이므로 수치 집계 방식이 다르다. Random Forest 원본 prediction은 동일하다.
