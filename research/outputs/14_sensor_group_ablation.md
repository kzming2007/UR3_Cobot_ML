# 14 센서 그룹 ablation 결과

## 고정 설계

- 사전 고정: `research/2026-08-23_sensor_group_ablation_preregistration.md`.
- 입력: cycle_run 경계를 넘지 않는 10-step×19개 원본 센서.
- 특징: 선택 센서마다 mean, std, min, max, range, delta, slope.
- 모델: SMOTE + Random Forest 300 trees, random_state 42, score > 0.50.
- 평가: 9개 acquisition block을 한 번씩 test로 사용.
- all_19 27개 fold는 기존 실험 12의 score와 prediction에 모두 일치했다.

## 핵심 결과

- `System_Failure` 전류 계열 제거: recall 1.0000->0.9712 (-0.0288), 오경보율 0.0394->0.0866 (+0.0472).
- `ProtectiveStop` 전류 계열 제거: recall 0.9737->0.9868 (+0.0132), 오경보율 0.0129->0.0581 (+0.0452).
- `GripLost` 전류 계열 제거: recall 0.9375->0.5417 (-0.3958), 오경보율 0.0273->0.0219 (-0.0055).
- `System_Failure` Tool current 제거: recall 1.0000->1.0000 (+0.0000), 오경보율 0.0394->0.0472 (+0.0079).
- `ProtectiveStop` Tool current 제거: recall 0.9737->0.9737 (+0.0000), 오경보율 0.0129->0.0323 (+0.0194).
- `GripLost` Tool current 제거: recall 0.9375->0.9375 (+0.0000), 오경보율 0.0273->0.0328 (+0.0055).
- Speed 단독은 event recall 0.9583-0.9868를 보였지만 정상 cycle 오경보율도 0.1097-0.2787였다.
- Temperature 단독 event recall은 0.0208-0.2308로 낮았다. 낮은 오경보와 함께 대부분의 event를 경보하지 않은 결과이므로 고장 탐지력이 높다고 해석하지 않는다.
- 종합하면 joint current는 System_Failure와 GripLost 구간 탐지에 중요한 정보를 제공하지만, 전류 계열이 모든 타깃에서 유일하거나 항상 최적인 것은 아니다. Tool current의 기여도 joint current와 분리해 해석해야 한다.

## All-sensor 기준선

| target | variant | feature_count | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | all_19 | 19 | 104 | 104 | 1.0 | 0.9644 | 1.0 | 1.0 | 127 | 5 | 0.0394 | 0.0169 | 0.0889 | 0.1111 | 0.8482 | 0.7702 | 0.9288 | 0.8515 |
| ProtectiveStop | all_19 | 19 | 76 | 74 | 0.9737 | 0.909 | 0.9928 | 0.5 | 155 | 2 | 0.0129 | 0.0035 | 0.0458 | 0.0952 | 0.8871 | 0.8056 | 0.9651 | 0.8706 |
| GripLost | all_19 | 19 | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.6667 | 183 | 5 | 0.0273 | 0.0117 | 0.0624 | 0.0952 | 0.8159 | 0.6659 | 0.9267 | 0.7742 |

## 센서 그룹 단독 사용

| target | variant | feature_count | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | current_family_only | 7 | 104 | 102 | 0.9808 | 0.9326 | 0.9947 | 0.8571 | 127 | 6 | 0.0472 | 0.0218 | 0.0992 | 0.1429 | 0.8505 | 0.7723 | 0.9254 | 0.8474 |
| System_Failure | joint_current_only | 6 | 104 | 103 | 0.9904 | 0.9475 | 0.9983 | 0.9524 | 127 | 7 | 0.0551 | 0.027 | 0.1094 | 0.2857 | 0.8507 | 0.7736 | 0.9244 | 0.846 |
| System_Failure | tool_current_only | 1 | 104 | 80 | 0.7692 | 0.6795 | 0.8397 | 0.6667 | 127 | 111 | 0.874 | 0.8051 | 0.9209 | 1.0 | 0.5192 | 0.269 | 0.5444 | 0.2765 |
| System_Failure | speed_only | 6 | 104 | 102 | 0.9808 | 0.9326 | 0.9947 | 0.8571 | 127 | 30 | 0.2362 | 0.1708 | 0.3172 | 0.3333 | 0.8029 | 0.7056 | 0.879 | 0.7674 |
| System_Failure | temperature_only | 6 | 104 | 24 | 0.2308 | 0.1603 | 0.3205 | 0.0 | 127 | 14 | 0.1102 | 0.0668 | 0.1765 | 0.8571 | 0.4892 | 0.1447 | 0.5563 | 0.2866 |
| ProtectiveStop | current_family_only | 7 | 76 | 73 | 0.9605 | 0.8903 | 0.9865 | 0.8889 | 155 | 9 | 0.0581 | 0.0308 | 0.1067 | 0.1304 | 0.8773 | 0.789 | 0.9675 | 0.8483 |
| ProtectiveStop | joint_current_only | 6 | 76 | 73 | 0.9605 | 0.8903 | 0.9865 | 0.8889 | 155 | 9 | 0.0581 | 0.0308 | 0.1067 | 0.1304 | 0.8739 | 0.7832 | 0.9655 | 0.8378 |
| ProtectiveStop | tool_current_only | 1 | 76 | 58 | 0.7632 | 0.6564 | 0.8446 | 0.5 | 155 | 109 | 0.7032 | 0.6271 | 0.7695 | 0.8095 | 0.5803 | 0.2845 | 0.6523 | 0.2263 |
| ProtectiveStop | speed_only | 6 | 76 | 75 | 0.9868 | 0.9292 | 0.9977 | 0.5 | 155 | 17 | 0.1097 | 0.0696 | 0.1686 | 0.2609 | 0.836 | 0.7218 | 0.9416 | 0.7856 |
| ProtectiveStop | temperature_only | 6 | 76 | 10 | 0.1316 | 0.0731 | 0.2255 | 0.0 | 155 | 9 | 0.0581 | 0.0308 | 0.1067 | 0.5 | 0.4807 | 0.0469 | 0.6151 | 0.1832 |
| GripLost | current_family_only | 7 | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.8 | 183 | 19 | 0.1038 | 0.0675 | 0.1565 | 0.28 | 0.819 | 0.6756 | 0.921 | 0.7429 |
| GripLost | joint_current_only | 6 | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.6667 | 183 | 24 | 0.1311 | 0.0897 | 0.1877 | 0.4 | 0.8149 | 0.6693 | 0.92 | 0.7471 |
| GripLost | tool_current_only | 1 | 48 | 31 | 0.6458 | 0.5044 | 0.7657 | 0.4444 | 183 | 140 | 0.765 | 0.6985 | 0.8206 | 0.8571 | 0.497 | 0.1225 | 0.484 | 0.1105 |
| GripLost | speed_only | 6 | 48 | 46 | 0.9583 | 0.8602 | 0.9885 | 0.8333 | 183 | 51 | 0.2787 | 0.2188 | 0.3477 | 0.4667 | 0.7516 | 0.5581 | 0.8276 | 0.5983 |
| GripLost | temperature_only | 6 | 48 | 1 | 0.0208 | 0.0037 | 0.109 | 0.0 | 183 | 1 | 0.0055 | 0.001 | 0.0303 | 0.0476 | 0.4712 | 0.0037 | 0.5137 | 0.124 |

## 센서 그룹 제거

| target | variant | feature_count | event_cycles | detected_event_cycles | event_cycle_recall | event_cycle_recall_ci95_low | event_cycle_recall_ci95_high | event_cycle_recall_min_block | normal_cycles | false_alarm_cycles | normal_cycle_false_alarm_rate | normal_cycle_false_alarm_rate_ci95_low | normal_cycle_false_alarm_rate_ci95_high | normal_cycle_false_alarm_rate_max_block | window_macro_f1 | window_positive_f1 | window_roc_auc | window_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | drop_current_family | 12 | 104 | 101 | 0.9712 | 0.9186 | 0.9901 | 0.8571 | 127 | 11 | 0.0866 | 0.0491 | 0.1485 | 0.1818 | 0.7847 | 0.6688 | 0.857 | 0.7233 |
| System_Failure | drop_speed | 13 | 104 | 102 | 0.9808 | 0.9326 | 0.9947 | 0.8571 | 127 | 5 | 0.0394 | 0.0169 | 0.0889 | 0.4286 | 0.8092 | 0.7041 | 0.911 | 0.8003 |
| System_Failure | drop_temperature | 13 | 104 | 104 | 1.0 | 0.9644 | 1.0 | 1.0 | 127 | 8 | 0.063 | 0.0323 | 0.1194 | 0.1667 | 0.8686 | 0.8012 | 0.9319 | 0.8685 |
| System_Failure | drop_tool_current | 18 | 104 | 104 | 1.0 | 0.9644 | 1.0 | 1.0 | 127 | 6 | 0.0472 | 0.0218 | 0.0992 | 0.1667 | 0.8461 | 0.7677 | 0.9273 | 0.8471 |
| ProtectiveStop | drop_current_family | 12 | 76 | 75 | 0.9868 | 0.9292 | 0.9977 | 0.5 | 155 | 9 | 0.0581 | 0.0308 | 0.1067 | 0.1429 | 0.8513 | 0.7449 | 0.9297 | 0.7881 |
| ProtectiveStop | drop_speed | 13 | 76 | 72 | 0.9474 | 0.8723 | 0.9793 | 0.3333 | 155 | 3 | 0.0194 | 0.0066 | 0.0554 | 0.25 | 0.868 | 0.7709 | 0.9619 | 0.8345 |
| ProtectiveStop | drop_temperature | 13 | 76 | 74 | 0.9737 | 0.909 | 0.9928 | 0.5 | 155 | 12 | 0.0774 | 0.0448 | 0.1304 | 0.3043 | 0.8882 | 0.8082 | 0.9694 | 0.8672 |
| ProtectiveStop | drop_tool_current | 18 | 76 | 74 | 0.9737 | 0.909 | 0.9928 | 0.5 | 155 | 5 | 0.0323 | 0.0139 | 0.0733 | 0.1429 | 0.8844 | 0.8012 | 0.9625 | 0.867 |
| GripLost | drop_current_family | 12 | 48 | 26 | 0.5417 | 0.4029 | 0.6742 | 0.0 | 183 | 4 | 0.0219 | 0.0085 | 0.0548 | 0.08 | 0.6446 | 0.3398 | 0.7758 | 0.4918 |
| GripLost | drop_speed | 13 | 48 | 44 | 0.9167 | 0.8045 | 0.9671 | 0.6667 | 183 | 9 | 0.0492 | 0.0261 | 0.0908 | 0.1875 | 0.7535 | 0.5501 | 0.9056 | 0.6877 |
| GripLost | drop_temperature | 13 | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.6667 | 183 | 10 | 0.0546 | 0.0299 | 0.0977 | 0.1333 | 0.8354 | 0.7041 | 0.9258 | 0.7907 |
| GripLost | drop_tool_current | 18 | 48 | 45 | 0.9375 | 0.8316 | 0.9785 | 0.6667 | 183 | 6 | 0.0328 | 0.0151 | 0.0697 | 0.1429 | 0.8221 | 0.6774 | 0.927 | 0.7806 |

## All-sensor 대비 변화량

| target | variant | event_cycle_recall_delta | normal_cycle_false_alarm_rate_delta | window_macro_f1_delta | window_pr_auc_delta |
| --- | --- | --- | --- | --- | --- |
| System_Failure | current_family_only | -0.0192 | 0.0079 | 0.0023 | -0.0041 |
| System_Failure | joint_current_only | -0.0096 | 0.0157 | 0.0025 | -0.0055 |
| System_Failure | tool_current_only | -0.2308 | 0.8346 | -0.3289 | -0.5749 |
| System_Failure | speed_only | -0.0192 | 0.1969 | -0.0453 | -0.0841 |
| System_Failure | temperature_only | -0.7692 | 0.0709 | -0.359 | -0.5648 |
| System_Failure | drop_current_family | -0.0288 | 0.0472 | -0.0635 | -0.1282 |
| System_Failure | drop_speed | -0.0192 | 0.0 | -0.039 | -0.0512 |
| System_Failure | drop_temperature | 0.0 | 0.0236 | 0.0204 | 0.017 |
| System_Failure | drop_tool_current | 0.0 | 0.0079 | -0.0021 | -0.0044 |
| ProtectiveStop | current_family_only | -0.0132 | 0.0452 | -0.0098 | -0.0222 |
| ProtectiveStop | joint_current_only | -0.0132 | 0.0452 | -0.0132 | -0.0327 |
| ProtectiveStop | tool_current_only | -0.2105 | 0.6903 | -0.3068 | -0.6443 |
| ProtectiveStop | speed_only | 0.0132 | 0.0968 | -0.0511 | -0.0849 |
| ProtectiveStop | temperature_only | -0.8421 | 0.0452 | -0.4064 | -0.6873 |
| ProtectiveStop | drop_current_family | 0.0132 | 0.0452 | -0.0358 | -0.0825 |
| ProtectiveStop | drop_speed | -0.0263 | 0.0065 | -0.0191 | -0.0361 |
| ProtectiveStop | drop_temperature | 0.0 | 0.0645 | 0.0011 | -0.0034 |
| ProtectiveStop | drop_tool_current | 0.0 | 0.0194 | -0.0027 | -0.0036 |
| GripLost | current_family_only | 0.0 | 0.0765 | 0.003 | -0.0313 |
| GripLost | joint_current_only | 0.0 | 0.1038 | -0.001 | -0.0271 |
| GripLost | tool_current_only | -0.2917 | 0.7377 | -0.3189 | -0.6637 |
| GripLost | speed_only | 0.0208 | 0.2514 | -0.0643 | -0.1759 |
| GripLost | temperature_only | -0.9167 | -0.0219 | -0.3448 | -0.6502 |
| GripLost | drop_current_family | -0.3958 | -0.0055 | -0.1713 | -0.2824 |
| GripLost | drop_speed | -0.0208 | 0.0219 | -0.0625 | -0.0865 |
| GripLost | drop_temperature | 0.0 | 0.0273 | 0.0195 | 0.0165 |
| GripLost | drop_tool_current | 0.0 | 0.0055 | 0.0061 | 0.0064 |

- Recall delta는 클수록 유리하고 정상 cycle 오경보율 delta는 작을수록 유리하다.
- Window 지표 delta와 cycle 지표 delta의 방향이 다르면 평가 단위에 따른 trade-off로 본다.

## All-sensor 대비 paired cycle 오류

| target | variant | error_type | eligible_cycles | all_19_error_count | variant_error_count | shared_error_count | new_variant_error_count | corrected_all_19_error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | current_family_only | event_miss | 104 | 0 | 2 | 0 | 2 | 0 |
| System_Failure | current_family_only | false_alarm | 127 | 5 | 6 | 1 | 5 | 4 |
| System_Failure | joint_current_only | event_miss | 104 | 0 | 1 | 0 | 1 | 0 |
| System_Failure | joint_current_only | false_alarm | 127 | 5 | 7 | 1 | 6 | 4 |
| System_Failure | tool_current_only | event_miss | 104 | 0 | 24 | 0 | 24 | 0 |
| System_Failure | tool_current_only | false_alarm | 127 | 5 | 111 | 5 | 106 | 0 |
| System_Failure | speed_only | event_miss | 104 | 0 | 2 | 0 | 2 | 0 |
| System_Failure | speed_only | false_alarm | 127 | 5 | 30 | 5 | 25 | 0 |
| System_Failure | temperature_only | event_miss | 104 | 0 | 80 | 0 | 80 | 0 |
| System_Failure | temperature_only | false_alarm | 127 | 5 | 14 | 0 | 14 | 5 |
| System_Failure | drop_current_family | event_miss | 104 | 0 | 3 | 0 | 3 | 0 |
| System_Failure | drop_current_family | false_alarm | 127 | 5 | 11 | 5 | 6 | 0 |
| System_Failure | drop_speed | event_miss | 104 | 0 | 2 | 0 | 2 | 0 |
| System_Failure | drop_speed | false_alarm | 127 | 5 | 5 | 1 | 4 | 4 |
| System_Failure | drop_temperature | event_miss | 104 | 0 | 0 | 0 | 0 | 0 |
| System_Failure | drop_temperature | false_alarm | 127 | 5 | 8 | 5 | 3 | 0 |
| System_Failure | drop_tool_current | event_miss | 104 | 0 | 0 | 0 | 0 | 0 |
| System_Failure | drop_tool_current | false_alarm | 127 | 5 | 6 | 5 | 1 | 0 |
| ProtectiveStop | current_family_only | event_miss | 76 | 2 | 3 | 1 | 2 | 1 |
| ProtectiveStop | current_family_only | false_alarm | 155 | 2 | 9 | 1 | 8 | 1 |
| ProtectiveStop | joint_current_only | event_miss | 76 | 2 | 3 | 1 | 2 | 1 |
| ProtectiveStop | joint_current_only | false_alarm | 155 | 2 | 9 | 1 | 8 | 1 |
| ProtectiveStop | tool_current_only | event_miss | 76 | 2 | 18 | 0 | 18 | 2 |
| ProtectiveStop | tool_current_only | false_alarm | 155 | 2 | 109 | 2 | 107 | 0 |
| ProtectiveStop | speed_only | event_miss | 76 | 2 | 1 | 1 | 0 | 1 |
| ProtectiveStop | speed_only | false_alarm | 155 | 2 | 17 | 2 | 15 | 0 |
| ProtectiveStop | temperature_only | event_miss | 76 | 2 | 66 | 2 | 64 | 0 |
| ProtectiveStop | temperature_only | false_alarm | 155 | 2 | 9 | 0 | 9 | 2 |
| ProtectiveStop | drop_current_family | event_miss | 76 | 2 | 1 | 1 | 0 | 1 |
| ProtectiveStop | drop_current_family | false_alarm | 155 | 2 | 9 | 2 | 7 | 0 |
| ProtectiveStop | drop_speed | event_miss | 76 | 2 | 4 | 2 | 2 | 0 |
| ProtectiveStop | drop_speed | false_alarm | 155 | 2 | 3 | 0 | 3 | 2 |
| ProtectiveStop | drop_temperature | event_miss | 76 | 2 | 2 | 2 | 0 | 0 |
| ProtectiveStop | drop_temperature | false_alarm | 155 | 2 | 12 | 2 | 10 | 0 |
| ProtectiveStop | drop_tool_current | event_miss | 76 | 2 | 2 | 2 | 0 | 0 |
| ProtectiveStop | drop_tool_current | false_alarm | 155 | 2 | 5 | 2 | 3 | 0 |
| GripLost | current_family_only | event_miss | 48 | 3 | 3 | 2 | 1 | 1 |
| GripLost | current_family_only | false_alarm | 183 | 5 | 19 | 3 | 16 | 2 |
| GripLost | joint_current_only | event_miss | 48 | 3 | 3 | 3 | 0 | 0 |
| GripLost | joint_current_only | false_alarm | 183 | 5 | 24 | 4 | 20 | 1 |
| GripLost | tool_current_only | event_miss | 48 | 3 | 17 | 2 | 15 | 1 |
| GripLost | tool_current_only | false_alarm | 183 | 5 | 140 | 3 | 137 | 2 |
| GripLost | speed_only | event_miss | 48 | 3 | 2 | 1 | 1 | 2 |
| GripLost | speed_only | false_alarm | 183 | 5 | 51 | 4 | 47 | 1 |
| GripLost | temperature_only | event_miss | 48 | 3 | 47 | 3 | 44 | 0 |
| GripLost | temperature_only | false_alarm | 183 | 5 | 1 | 0 | 1 | 5 |
| GripLost | drop_current_family | event_miss | 48 | 3 | 22 | 2 | 20 | 1 |
| GripLost | drop_current_family | false_alarm | 183 | 5 | 4 | 0 | 4 | 5 |
| GripLost | drop_speed | event_miss | 48 | 3 | 4 | 3 | 1 | 0 |
| GripLost | drop_speed | false_alarm | 183 | 5 | 9 | 3 | 6 | 2 |
| GripLost | drop_temperature | event_miss | 48 | 3 | 3 | 3 | 0 | 0 |
| GripLost | drop_temperature | false_alarm | 183 | 5 | 10 | 4 | 6 | 1 |
| GripLost | drop_tool_current | event_miss | 48 | 3 | 3 | 3 | 0 | 0 |
| GripLost | drop_tool_current | false_alarm | 183 | 5 | 6 | 3 | 3 | 2 |

## 사전 고정 해석 규칙 적용

- `System_Failure` 단독 그룹: current_family_only가 speed_only와 temperature_only를 두 cycle 지표에서 각각 Pareto 우세했다.
- `System_Failure` `drop_current_family`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `System_Failure` `drop_speed`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `System_Failure` `drop_temperature`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `System_Failure` `drop_tool_current`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `ProtectiveStop` 단독 그룹: current_family_only가 speed_only와 temperature_only를 모두 Pareto 우세하지는 않아 전류 단독 우위를 일반화할 수 없다.
- `ProtectiveStop` `drop_current_family`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `ProtectiveStop` `drop_speed`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `ProtectiveStop` `drop_temperature`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `ProtectiveStop` `drop_tool_current`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `GripLost` 단독 그룹: current_family_only가 speed_only와 temperature_only를 모두 Pareto 우세하지는 않아 전류 단독 우위를 일반화할 수 없다.
- `GripLost` `drop_current_family`: Recall과 오경보 방향이 엇갈리는 trade-off다.
- `GripLost` `drop_speed`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `GripLost` `drop_temperature`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- `GripLost` `drop_tool_current`: all_19가 recall과 오경보 기준에서 기술적으로 우세하다.
- Pareto 우세는 recall과 오경보의 방향을 함께 본 기술적 비교이며 통계적 유의성이나 인과관계를 뜻하지 않는다.

## 실행 범위

- Random Forest run: 270개.
- Window prediction: 140,010개.
- 기록된 학습 시간 합: 373.6초.

## 해석 제한

- Feature 수와 SMOTE 공간이 variant마다 달라 제거 효과를 순수한 인과 기여로 해석할 수 없다.
- Tool current의 구간 탐지 성능은 고장 발생 전 유용성을 뜻하지 않는다.
- Temperature 성능에는 thermal/session drift가 포함될 수 있다.
- 4,667개 window는 겹치며 독립 평가 단위는 202개 cycle이다.
- 9개 block은 실제 공정조건 정답이 아닌 수집 구간 proxy다.
- 단일 공개 데이터와 Random Forest 고정 seed 1회의 내부 비교다.
