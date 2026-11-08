# Fault-context alert metric correction

## 목적

기존 결과의 `normal_cycle_false_alarm_rate`는 실제로 각 target이 없는 모든 cycle을 분모로 사용했다. 따라서 `ProtectiveStop` 평가에는 `GripLost`만 발생한 cycle이, `GripLost` 평가에는 `ProtectiveStop`만 발생한 cycle이 포함됐다. 이 분석은 재학습이나 threshold 변경 없이 기존 cycle 결과를 다음 세 범주로 분리한다.

- `target_negative_alert_rate`: 해당 target이 없는 모든 cycle에서의 경보율
- `true_normal_false_alarm_rate`: 어떤 고장도 없는 cycle에서의 오경보율
- `cross_fault_alert_rate`: 해당 target은 없지만 다른 고장만 있는 cycle에서의 경보율

`System_Failure`는 두 개별 고장의 합집합이므로 교차 고장 범주가 없다.

## 모델 비교

| candidate | target | event_cycle_recall | target_negative_alert_rate | true_normal_false_alarm_rate | cross_fault_alert_rate | other_fault_only_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 1d_cnn_19_raw | System_Failure | 1.0 | 0.1811 | 0.1811 |  | 0 |
| 1d_cnn_19_raw | ProtectiveStop | 0.9605 | 0.1355 | 0.1339 | 0.1429 | 28 |
| 1d_cnn_19_raw | GripLost | 1.0 | 0.2678 | 0.2126 | 0.3929 | 56 |
| lstm_autoencoder_19_raw_q95 | System_Failure | 0.0481 | 0.1024 | 0.1024 |  | 0 |
| lstm_autoencoder_19_raw_q95 | ProtectiveStop | 0.0658 | 0.0839 | 0.1024 | 0.0 | 28 |
| lstm_autoencoder_19_raw_q95 | GripLost | 0.0 | 0.1038 | 0.1024 | 0.1071 | 56 |
| rf_19_raw_window | System_Failure | 1.0 | 0.0394 | 0.0394 |  | 0 |
| rf_19_raw_window | ProtectiveStop | 0.9737 | 0.0129 | 0.0157 | 0.0 | 28 |
| rf_19_raw_window | GripLost | 0.9375 | 0.0273 | 0.0157 | 0.0536 | 56 |
| logistic_regression | System_Failure | 0.9904 | 0.4567 | 0.4567 |  | 0 |
| logistic_regression | ProtectiveStop | 0.9737 | 0.271 | 0.2677 | 0.2857 | 28 |
| logistic_regression | GripLost | 1.0 | 0.541 | 0.4724 | 0.6964 | 56 |
| rbf_svm | System_Failure | 0.9904 | 0.0787 | 0.0787 |  | 0 |
| rbf_svm | ProtectiveStop | 0.9605 | 0.0065 | 0.0 | 0.0357 | 28 |
| rbf_svm | GripLost | 0.9583 | 0.1585 | 0.0945 | 0.3036 | 56 |

## 센서 그룹 ablation

| candidate | target | event_cycle_recall | target_negative_alert_rate | true_normal_false_alarm_rate | cross_fault_alert_rate | other_fault_only_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| all_19 | System_Failure | 1.0 | 0.0394 | 0.0394 |  | 0 |
| all_19 | ProtectiveStop | 0.9737 | 0.0129 | 0.0157 | 0.0 | 28 |
| all_19 | GripLost | 0.9375 | 0.0273 | 0.0157 | 0.0536 | 56 |
| current_family_only | System_Failure | 0.9808 | 0.0472 | 0.0472 |  | 0 |
| current_family_only | ProtectiveStop | 0.9605 | 0.0581 | 0.0551 | 0.0714 | 28 |
| current_family_only | GripLost | 0.9375 | 0.1038 | 0.0394 | 0.25 | 56 |
| drop_current_family | System_Failure | 0.9712 | 0.0866 | 0.0866 |  | 0 |
| drop_current_family | ProtectiveStop | 0.9868 | 0.0581 | 0.0394 | 0.1429 | 28 |
| drop_current_family | GripLost | 0.5417 | 0.0219 | 0.0079 | 0.0536 | 56 |
| drop_speed | System_Failure | 0.9808 | 0.0394 | 0.0394 |  | 0 |
| drop_speed | ProtectiveStop | 0.9474 | 0.0194 | 0.0236 | 0.0 | 28 |
| drop_speed | GripLost | 0.9167 | 0.0492 | 0.0236 | 0.1071 | 56 |
| drop_temperature | System_Failure | 1.0 | 0.063 | 0.063 |  | 0 |
| drop_temperature | ProtectiveStop | 0.9737 | 0.0774 | 0.0551 | 0.1786 | 28 |
| drop_temperature | GripLost | 0.9375 | 0.0546 | 0.0315 | 0.1071 | 56 |
| drop_tool_current | System_Failure | 1.0 | 0.0472 | 0.0472 |  | 0 |
| drop_tool_current | ProtectiveStop | 0.9737 | 0.0323 | 0.0236 | 0.0714 | 28 |
| drop_tool_current | GripLost | 0.9375 | 0.0328 | 0.0079 | 0.0893 | 56 |
| joint_current_only | System_Failure | 0.9904 | 0.0551 | 0.0551 |  | 0 |
| joint_current_only | ProtectiveStop | 0.9605 | 0.0581 | 0.0551 | 0.0714 | 28 |
| joint_current_only | GripLost | 0.9375 | 0.1311 | 0.0787 | 0.25 | 56 |
| speed_only | System_Failure | 0.9808 | 0.2362 | 0.2362 |  | 0 |
| speed_only | ProtectiveStop | 0.9868 | 0.1097 | 0.0709 | 0.2857 | 28 |
| speed_only | GripLost | 0.9583 | 0.2787 | 0.2205 | 0.4107 | 56 |
| temperature_only | System_Failure | 0.2308 | 0.1102 | 0.1102 |  | 0 |
| temperature_only | ProtectiveStop | 0.1316 | 0.0581 | 0.0551 | 0.0714 | 28 |
| temperature_only | GripLost | 0.0208 | 0.0055 | 0.0 | 0.0179 | 56 |
| tool_current_only | System_Failure | 0.7692 | 0.874 | 0.874 |  | 0 |
| tool_current_only | ProtectiveStop | 0.7632 | 0.7032 | 0.6929 | 0.75 | 28 |
| tool_current_only | GripLost | 0.6458 | 0.765 | 0.7323 | 0.8393 | 56 |

## 해석 제한

- 교차 고장 경보는 곧바로 오류라고 단정할 수 없다. 두 고장 유형이 일부 센서 패턴을 공유할 수 있기 때문이다.
- 반대로 target별 분류 성능으로 보고할 때는 교차 고장 경보를 해당 target의 정답으로 계산할 수 없다.
- 세 지표는 동일한 cycle 예측을 서로 다른 운영 질문에 맞춰 재집계한 값이며 독립적인 새 실험이 아니다.
- 현재 결과는 동일 데이터셋의 block-held-out 내부 검증이며 외부 로봇·외부 수집 세션 일반화 증거가 아니다.
