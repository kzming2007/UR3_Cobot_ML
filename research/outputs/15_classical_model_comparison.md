# 15 동일 시계열 특징 기반 기본 분류 모델 비교

## 고정 설계

- 사전 고정: `research/2026-08-23_classical_model_comparison_preregistration.md`.
- 입력: cycle_run 경계를 넘지 않는 10-step×19개 원본 센서의 133개 통계 특징.
- 모델: Logistic Regression L2 C=1, RBF SVM C=1·gamma=scale, 기존 Random Forest 300 trees.
- 전처리: 기본 모델은 학습 fold 내부 StandardScaler와 SMOTE, Random Forest는 기존 SMOTE 결과 재사용.
- 평가: 9개 acquisition block을 한 번씩 test로 사용하고 score > 0.50 적용.
- 새 기본 모델 학습: 54회, 기록된 실행 시간 합 852.3초.

## Primary cycle 결과와 secondary window 결과

| target | model | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | random_forest | 104 | 104 | 1.0 | 0.9644 | 1.0 | 1.0 | 127 | 5 | 0.0394 | 0.0169 | 0.0889 | 0.1111 | 0.8482 | 0.7702 | 0.9288 | 0.8515 |
| System_Failure | logistic_regression | 104 | 103 | 0.9904 | 0.9475 | 0.9983 | 0.9524 | 127 | 58 | 0.4567 | 0.3726 | 0.5433 | 0.7222 | 0.7044 | 0.5839 | 0.8105 | 0.6181 |
| System_Failure | rbf_svm | 104 | 103 | 0.9904 | 0.9475 | 0.9983 | 0.8571 | 127 | 10 | 0.0787 | 0.0433 | 0.1389 | 0.4286 | 0.8234 | 0.731 | 0.9139 | 0.8141 |
| ProtectiveStop | random_forest | 76 | 74 | 0.9737 | 0.909 | 0.9928 | 0.5 | 155 | 2 | 0.0129 | 0.0035 | 0.0458 | 0.0952 | 0.8871 | 0.8056 | 0.9651 | 0.8706 |
| ProtectiveStop | logistic_regression | 76 | 74 | 0.9737 | 0.909 | 0.9928 | 0.875 | 155 | 42 | 0.271 | 0.2072 | 0.3458 | 0.5 | 0.8034 | 0.6738 | 0.9136 | 0.6971 |
| ProtectiveStop | rbf_svm | 76 | 73 | 0.9605 | 0.8903 | 0.9865 | 0.5 | 155 | 1 | 0.0065 | 0.0011 | 0.0356 | 0.05 | 0.8614 | 0.7588 | 0.9576 | 0.8231 |
| GripLost | random_forest | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.6667 | 183 | 5 | 0.0273 | 0.0117 | 0.0624 | 0.0952 | 0.8159 | 0.6659 | 0.9267 | 0.7742 |
| GripLost | logistic_regression | 48 | 48 | 1.0 | 0.9259 | 1.0 | 1.0 | 183 | 99 | 0.541 | 0.4687 | 0.6116 | 0.92 | 0.7091 | 0.5165 | 0.875 | 0.5571 |
| GripLost | rbf_svm | 48 | 46 | 0.9583 | 0.8602 | 0.9885 | 0.6667 | 183 | 29 | 0.1585 | 0.1127 | 0.2183 | 0.44 | 0.7958 | 0.6354 | 0.9203 | 0.7147 |

## Random Forest 대비 해석

- `System_Failure` `logistic_regression`: event recall 0.9904 (-0.0096), 정상 cycle 오경보율 0.4567 (+0.4173); Random Forest가 두 cycle 지표에서 기술적으로 우세했다.
- `System_Failure` `rbf_svm`: event recall 0.9904 (-0.0096), 정상 cycle 오경보율 0.0787 (+0.0394); Random Forest가 두 cycle 지표에서 기술적으로 우세했다.
- `ProtectiveStop` `logistic_regression`: event recall 0.9737 (+0.0000), 정상 cycle 오경보율 0.2710 (+0.2581); Random Forest가 두 cycle 지표에서 기술적으로 우세했다.
- `ProtectiveStop` `rbf_svm`: event recall 0.9605 (-0.0132), 정상 cycle 오경보율 0.0065 (-0.0065); recall과 오경보 방향이 엇갈리는 trade-off였다.
- `GripLost` `logistic_regression`: event recall 1.0000 (+0.0625), 정상 cycle 오경보율 0.5410 (+0.5137); recall과 오경보 방향이 엇갈리는 trade-off였다.
- `GripLost` `rbf_svm`: event recall 0.9583 (+0.0208), 정상 cycle 오경보율 0.1585 (+0.1311); recall과 오경보 방향이 엇갈리는 trade-off였다.

## Paired cycle 오류

| target | model | error_type | eligible_cycles | random_forest_error_count | model_error_count | shared_error_count | new_model_error_count | corrected_rf_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | logistic_regression | event_miss | 104 | 0 | 1 | 0 | 1 | 0 |
| System_Failure | logistic_regression | false_alarm | 127 | 5 | 58 | 3 | 55 | 2 |
| System_Failure | rbf_svm | event_miss | 104 | 0 | 1 | 0 | 1 | 0 |
| System_Failure | rbf_svm | false_alarm | 127 | 5 | 10 | 1 | 9 | 4 |
| ProtectiveStop | logistic_regression | event_miss | 76 | 2 | 2 | 1 | 1 | 1 |
| ProtectiveStop | logistic_regression | false_alarm | 155 | 2 | 42 | 2 | 40 | 0 |
| ProtectiveStop | rbf_svm | event_miss | 76 | 2 | 3 | 2 | 1 | 0 |
| ProtectiveStop | rbf_svm | false_alarm | 155 | 2 | 1 | 0 | 1 | 2 |
| GripLost | logistic_regression | event_miss | 48 | 3 | 0 | 0 | 0 | 3 |
| GripLost | logistic_regression | false_alarm | 183 | 5 | 99 | 4 | 95 | 1 |
| GripLost | rbf_svm | event_miss | 48 | 3 | 2 | 2 | 0 | 1 |
| GripLost | rbf_svm | false_alarm | 183 | 5 | 29 | 2 | 27 | 3 |

- Event miss는 실제 positive window를 하나도 잡지 못한 event cycle이다.
- False alarm은 정상 cycle의 window 중 하나 이상에서 경보한 경우다.
- `new_model_error_count`는 Random Forest가 맞혔지만 비교 모델이 틀린 cycle이다.
- `corrected_rf_error_count`는 Random Forest 오류를 비교 모델이 바로잡은 cycle이다.

## 해석 제한

- 같은 입력과 split을 사용했지만 Logistic Regression과 SVM에는 모델에 필요한 표준화를 적용했다.
- Hyperparameter와 threshold를 탐색하지 않았으므로 각 모델 계열의 최고 성능 비교가 아니다.
- SVM probability와 Random Forest probability는 산출 방식이 다르다.
- Cycle 지표를 우선하며 window 지표만 좋아진 경우 운영상 우위로 해석하지 않는다.
- 구간 탐지 결과이며 조기 고장 예측 또는 외부 일반화의 근거가 아니다.
- 이 결과를 본 뒤 모델, C, kernel, gamma 또는 threshold를 추가로 조정하지 않는다.
