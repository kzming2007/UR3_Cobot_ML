# 22 Post-hoc checks from saved results

Post-hoc and descriptive. No model was trained.

## row_window_prevalence

| target | feature_level | window_size | pr_auc | positive_rate | pr_auc_over_rate |
| --- | --- | --- | --- | --- | --- |
| System_Failure | row_baseline |  | 0.643 | 0.0704 | 9.1299 |
| System_Failure | best_window_feature | 20.0 | 0.8927 | 0.4491 | 1.9878 |
| ProtectiveStop | row_baseline |  | 0.7409 | 0.0378 | 19.6025 |
| ProtectiveStop | best_window_feature | 5.0 | 0.8796 | 0.1024 | 8.5923 |
| GripLost | row_baseline |  | 0.5604 | 0.0328 | 17.0871 |
| GripLost | best_window_feature | 10.0 | 0.7831 | 0.1117 | 7.0137 |

## raw_block_counts

| block | raw_cycle_runs |
| --- | --- |
| 1 | 25 |
| 2 | 25 |
| 3 | 24 |
| 4 | 25 |
| 5 | 25 |
| 6 | 4 |
| 7 | 25 |
| 8 | 24 |
| 9 | 27 |
| 10 | 31 |
| 11 | 14 |

## block_composition

| block | cycles | normal | system_failure | protective_stop | grip_lost |
| --- | --- | --- | --- | --- | --- |
| 1 | 25 | 18 | 7 | 2 | 5 |
| 2 | 25 | 20 | 5 | 2 | 3 |
| 3 | 24 | 18 | 6 | 3 | 5 |
| 4 | 25 | 12 | 13 | 5 | 9 |
| 5 | 25 | 19 | 6 | 5 | 1 |
| 7 | 25 | 16 | 9 | 8 | 4 |
| 8 | 24 | 11 | 13 | 10 | 9 |
| 9 | 27 | 6 | 21 | 18 | 6 |
| 10 | 31 | 7 | 24 | 23 | 6 |

## data_gaps

| target | windows | cycles | spliced_windows | spliced_positive_windows | windows_with_60s_gap | positive_windows_with_positive_last_step | positive_windows | rows_missing_label_in_blocks | normal_cycles_with_missing_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 4667 | 231 | 307 | 93 | 3 | 267 | 1163 | 45 | 15 |
| ProtectiveStop | 4667 | 231 | 307 | 32 | 3 | 74 | 653 | 45 | 15 |
| GripLost | 4667 | 231 | 307 | 66 | 3 | 196 | 535 | 45 | 15 |

## same_false_alarm

| target | fault_cycles | rf_detected | rf_false_alarms | cnn_detected | cnn_false_alarms | cnn_cut_at_rf_false_alarms | cnn_false_alarms_at_cut | cnn_detected_at_cut | rf_cycle_auc | cnn_cycle_auc | rf_only_alarms | cnn_only_alarms | mcnemar_exact_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 104 | 104 | 5 | 104 | 23 | 0.7028 | 5 | 101 | 0.9976 | 0.9951 | 3 | 21 | 0.0003 |
| ProtectiveStop | 76 | 74 | 2 | 73 | 17 | 0.7465 | 2 | 68 | 0.9988 | 0.9747 | 2 | 17 | 0.0007 |
| GripLost | 48 | 45 | 2 | 48 | 27 | 0.7421 | 2 | 37 | 0.9965 | 0.9788 | 1 | 26 | 0.0 |

## missing_sensitivity

| model | target | fault_cycles | no_splice_detected | no_splice_false_alarms | normal_cycles | faults_only_in_spliced_windows | false_alarms_without_missing_label_cycles | normal_cycles_without_missing_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | System_Failure | 104 | 102 | 4 | 127 | 2 | 4 | 112 |
| random_forest | ProtectiveStop | 76 | 72 | 2 | 127 | 2 | 2 | 112 |
| random_forest | GripLost | 48 | 42 | 2 | 127 | 2 | 2 | 112 |
| 1d_cnn | System_Failure | 104 | 101 | 23 | 127 | 2 | 19 | 112 |
| 1d_cnn | ProtectiveStop | 76 | 70 | 16 | 127 | 2 | 13 | 112 |
| 1d_cnn | GripLost | 48 | 45 | 27 | 127 | 2 | 23 | 112 |

## ae_movement

| target | windows | spearman_current_std | spearman_speed_std | median_score_low_speed | median_score_rest | speed_std_positive | speed_std_negative | current_std_positive | current_std_negative | positive_share_in_low_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 4667 | 0.7858 | 0.4795 | 0.2079 | 0.6525 | 0.1674 | 0.2928 | 0.3738 | 0.4677 | 0.4445 |
| ProtectiveStop | 4667 | 0.7858 | 0.4795 | 0.2079 | 0.6525 | 0.0984 | 0.2899 | 0.477 | 0.4437 | 0.6279 |
| GripLost | 4667 | 0.7858 | 0.4795 | 0.2079 | 0.6525 | 0.2617 | 0.2617 | 0.2922 | 0.4693 | 0.2299 |

## pre_failure_baseline

| target | window_size | horizon | splits | positive_rate | all_positive_f1 | positive_f1 | runs_with_true_positive | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 5 | 10 | 30 | 0.0762 | 0.1416 | 0.1178 | 29 | 0.2389 | 0.7998 |
| ProtectiveStop | 10 | 3 | 30 | 0.0106 | 0.0209 | 0.1344 | 6 | 0.5117 | 0.8287 |
| GripLost | 5 | 3 | 30 | 0.0186 | 0.0365 | 0.3758 | 30 | 0.4585 | 0.9359 |
