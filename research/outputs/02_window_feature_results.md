# 02 Window Feature Baseline

## 범위

- Focus: 분류 기반 이상탐지.
- Target: `System_Failure`, `ProtectiveStop`, `GripLost`.
- Window: 시간상 연속한 `cycle_run` 경계를 넘지 않는 5, 10, 20 step sliding window.
- Label: window 내부에 positive target이 하나라도 있으면 positive.
- Feature: mean, std, min, max, range, first-last delta, simple slope.
- Split: 같은 `cycle_run`이 train/test에 동시에 들어가지 않는 `cycle_run_group` split.
- Models: class weighting을 적용한 Random Forest, SMOTE + Random Forest.

## Window 데이터 요약

| target | window_size | windows | cycle_runs | cycle_ids | positive_windows | positive_rate | mean_positive_points_in_positive_window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | 6359 | 249 | 240 | 1141 | 0.1794 | 2.1989 |
| System_Failure | 10 | 5114 | 249 | 240 | 1288 | 0.2519 | 2.8734 |
| System_Failure | 20 | 2632 | 248 | 239 | 1182 | 0.4491 | 3.7217 |
| ProtectiveStop | 5 | 6359 | 249 | 240 | 651 | 0.1024 | 2.0753 |
| ProtectiveStop | 10 | 5114 | 249 | 240 | 743 | 0.1453 | 2.6703 |
| ProtectiveStop | 20 | 2632 | 248 | 239 | 810 | 0.3078 | 3.0222 |
| GripLost | 5 | 6359 | 249 | 240 | 502 | 0.0789 | 2.3367 |
| GripLost | 10 | 5114 | 249 | 240 | 571 | 0.1117 | 3.0578 |
| GripLost | 20 | 2632 | 248 | 239 | 454 | 0.1725 | 4.3568 |

## 결과

| target | window_size | model | test_windows | positive_count | positive_rate | accuracy | macro_f1 | positive_recall | positive_precision | positive_f1 | roc_auc | pr_auc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | rf_plain | 1280 | 227 | 0.1773 | 0.9195 | 0.8354 | 0.5771 | 0.9493 | 0.7178 | 0.9464 | 0.8621 | 1046 | 7 | 96 | 131 |
| System_Failure | 5 | rf_smote | 1280 | 227 | 0.1773 | 0.9328 | 0.8766 | 0.7269 | 0.873 | 0.7933 | 0.9471 | 0.8699 | 1029 | 24 | 62 | 165 |
| System_Failure | 10 | rf_plain | 1030 | 257 | 0.2495 | 0.8874 | 0.8382 | 0.6732 | 0.8439 | 0.7489 | 0.9335 | 0.8671 | 741 | 32 | 84 | 173 |
| System_Failure | 10 | rf_smote | 1030 | 257 | 0.2495 | 0.8951 | 0.8632 | 0.8249 | 0.7709 | 0.797 | 0.936 | 0.8657 | 710 | 63 | 45 | 212 |
| System_Failure | 20 | rf_plain | 425 | 150 | 0.3529 | 0.8682 | 0.8558 | 0.8133 | 0.8133 | 0.8133 | 0.9407 | 0.8974 | 247 | 28 | 28 | 122 |
| System_Failure | 20 | rf_smote | 425 | 150 | 0.3529 | 0.8965 | 0.886 | 0.84 | 0.863 | 0.8514 | 0.9421 | 0.8927 | 255 | 20 | 24 | 126 |
| ProtectiveStop | 5 | rf_plain | 1280 | 133 | 0.1039 | 0.9547 | 0.857 | 0.6165 | 0.9213 | 0.7387 | 0.985 | 0.8754 | 1140 | 7 | 51 | 82 |
| ProtectiveStop | 5 | rf_smote | 1280 | 133 | 0.1039 | 0.9555 | 0.8776 | 0.7594 | 0.8016 | 0.7799 | 0.9843 | 0.8796 | 1122 | 25 | 32 | 101 |
| ProtectiveStop | 10 | rf_plain | 1030 | 148 | 0.1437 | 0.934 | 0.8569 | 0.6959 | 0.8175 | 0.7518 | 0.9741 | 0.8757 | 859 | 23 | 45 | 103 |
| ProtectiveStop | 10 | rf_smote | 1030 | 148 | 0.1437 | 0.9447 | 0.8915 | 0.8514 | 0.7826 | 0.8155 | 0.9714 | 0.8595 | 847 | 35 | 22 | 126 |
| ProtectiveStop | 20 | rf_plain | 425 | 102 | 0.24 | 0.8918 | 0.8352 | 0.6373 | 0.8784 | 0.7386 | 0.9009 | 0.8309 | 314 | 9 | 37 | 65 |
| ProtectiveStop | 20 | rf_smote | 425 | 102 | 0.24 | 0.9082 | 0.8683 | 0.7451 | 0.8539 | 0.7958 | 0.9014 | 0.8469 | 310 | 13 | 26 | 76 |
| GripLost | 5 | rf_plain | 1280 | 96 | 0.075 | 0.9562 | 0.7826 | 0.4167 | 1.0 | 0.5882 | 0.916 | 0.7191 | 1184 | 0 | 56 | 40 |
| GripLost | 5 | rf_smote | 1280 | 96 | 0.075 | 0.9578 | 0.8157 | 0.5312 | 0.85 | 0.6538 | 0.9175 | 0.7272 | 1175 | 9 | 45 | 51 |
| GripLost | 10 | rf_plain | 1030 | 116 | 0.1126 | 0.932 | 0.7681 | 0.4052 | 0.9792 | 0.5732 | 0.929 | 0.7761 | 913 | 1 | 69 | 47 |
| GripLost | 10 | rf_smote | 1030 | 116 | 0.1126 | 0.9417 | 0.8244 | 0.5517 | 0.8889 | 0.6809 | 0.9343 | 0.7831 | 906 | 8 | 52 | 64 |
| GripLost | 20 | rf_plain | 425 | 62 | 0.1459 | 0.8918 | 0.7393 | 0.4355 | 0.7105 | 0.54 | 0.8817 | 0.666 | 352 | 11 | 35 | 27 |
| GripLost | 20 | rf_smote | 425 | 62 | 0.1459 | 0.9035 | 0.7843 | 0.5484 | 0.7234 | 0.6239 | 0.9062 | 0.7086 | 350 | 13 | 28 | 34 |

## Row Baseline 대비 비교

| target | feature_level | window_size | model | macro_f1 | positive_recall | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| System_Failure | row_baseline |  | rf_smote | 0.7901 | 0.5638 | 0.643 |
| System_Failure | best_window_feature | 20 | rf_smote | 0.886 | 0.84 | 0.8927 |
| ProtectiveStop | row_baseline |  | rf_smote | 0.8428 | 0.7111 | 0.7409 |
| ProtectiveStop | best_window_feature | 5 | rf_smote | 0.8776 | 0.7594 | 0.8796 |
| GripLost | row_baseline |  | rf_smote | 0.7623 | 0.4898 | 0.5604 |
| GripLost | best_window_feature | 10 | rf_smote | 0.8244 | 0.5517 | 0.7831 |

## 해석 메모

- 이 실험은 cycle 전체를 하나의 표본으로 쓰지 않고, cycle 내부의 짧은 구간에서 시계열 통계 feature를 만든다.
- `cycle_run_group` split을 사용하므로 같은 연속 시행에서 만들어진 window가 train/test에 동시에 섞이지 않는다.
- 원본 `cycle` ID가 시간상 떨어진 구간에 재등장하더라도 서로 다른 `cycle_run`으로 처리한다.
- window label은 구간 단위 이상탐지 정의이므로, 고장 발생 전 예측 문제로 해석하지 않는다.
- `System_Failure`는 전체 이상탐지 타깃이고, `ProtectiveStop`과 `GripLost`는 고장 유형별 패턴 차이를 보기 위한 보조 타깃이다.
