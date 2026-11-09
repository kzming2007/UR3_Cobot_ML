# 25 Post-hoc checks of how far the block results reach

Added after the results were seen. None replaces a main table.

## recording_time

| dates | date | rows_without_time | nonpositive_gaps | largest_gap_hours | gap_row | gap_cycle | gap_previous_cycle | second_gap_seconds | largest_block_boundary_gap_seconds | temperature_first_block | temperature_last_block | largest_cycle_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-10-26 | 0 | 0 | 5.2261 | 2869 | 105 | 105 | 31.15 | 2.01 | 32.1072 | 41.883 | 264 |

## forward_split

| inputs | target | train_cycles | train_fault_cycles | detected | fault_cycles | false_alarms | normal_cycles | cycle_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all sensors | System_Failure | 103 | 32 | 53 | 71 | 0 | 56 | 0.9935 |
| all sensors | ProtectiveStop | 103 | 13 | 7 | 63 | 0 | 56 | 0.9984 |
| all sensors | GripLost | 103 | 22 | 0 | 25 | 0 | 56 | 0.9832 |
| no temperature | System_Failure | 103 | 32 | 70 | 71 | 2 | 56 | 0.997 |
| no temperature | ProtectiveStop | 103 | 13 | 55 | 63 | 0 | 56 | 1.0 |
| no temperature | GripLost | 103 | 22 | 22 | 25 | 5 | 56 | 0.9921 |

## cycle_length

| target | median_rows_fault | median_rows_normal | cut | detected | fault_cycles | false_alarms | normal_cycles | cross_alerts | other_fault_cycles | cycle_auc | train_cut_detected | train_cut_false_alarms | rf_median_alert_rows_after_fault | rf_alert_cycles | rf_alerts_before_fault |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | 29.0 | 24.0 | 27 | 82 | 104 | 1 | 127 | 0 | 0 | 0.8722 | 82 | 1 | 5.0 | 104 | 6 |
| ProtectiveStop | 32.5 | 24.0 | 27 | 76 | 76 | 1 | 127 | 6 | 28 | 0.9938 | 76 | 1 | 5.0 | 74 | 0 |
| GripLost | 27.5 | 24.0 | 27 | 26 | 48 | 1 | 127 | 56 | 56 | 0.7304 | 26 | 1 | 0.0 | 45 | 12 |

## same_folds

| target | level | units | cycles | random_macro_f1 | cycle_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| System_Failure | rows | 7355 | 249 | 0.787 | 0.7239 |
| System_Failure | windows | 4667 | 231 | 0.9459 | 0.85 |
| ProtectiveStop | rows | 7355 | 249 | 0.8169 | 0.7648 |
| ProtectiveStop | windows | 4667 | 231 | 0.9609 | 0.8918 |
| GripLost | rows | 7355 | 249 | 0.7562 | 0.6843 |
| GripLost | windows | 4667 | 231 | 0.9405 | 0.8157 |

## autoencoder

| epochs | target | detected | fault_cycles | false_alarms | normal_cycles | cross_alerts | other_fault_cycles | window_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at most 40 | System_Failure | 5 | 104 | 13 | 127 | 0 | 0 | 0.4202 |
| 400 | System_Failure | 46 | 104 | 11 | 127 | 0 | 0 | 0.8053 |
| at most 40 | ProtectiveStop | 5 | 76 | 13 | 127 | 0 | 28 | 0.4599 |
| 400 | ProtectiveStop | 40 | 76 | 11 | 127 | 5 | 28 | 0.8969 |
| at most 40 | GripLost | 0 | 48 | 13 | 127 | 6 | 56 | 0.3934 |
| 400 | GripLost | 8 | 48 | 11 | 127 | 30 | 56 | 0.607 |

## autoencoder_fit

| test_block | validation_block | saved_loss_min | saved_loss_max | window_mean_loss | ae_runs_at_cap | ae_runs | cnn_runs_at_cap | cnn_runs | fixed_400_train_mse_mean | fixed_400_calibration_normal_mse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 17.1847 | 17.4793 | 0.6447 | 3 | 3 | 0 | 9 | 0.1476 | 0.2859 |
| 2 | 3 | 0.7571 | 0.7641 | 0.6942 | 0 | 3 | 0 | 9 | 0.133 | 0.1723 |
| 3 | 4 | 0.6857 | 0.6922 | 0.6438 | 0 | 3 | 0 | 9 | 0.1325 | 0.1873 |
| 4 | 5 | 0.7214 | 0.7271 | 0.6791 | 3 | 3 | 1 | 9 | 0.1212 | 0.1395 |
| 5 | 7 | 0.6921 | 0.6948 | 0.6533 | 3 | 3 | 0 | 9 | 0.1142 | 0.1427 |
| 7 | 8 | 0.6354 | 0.6364 | 0.6088 | 3 | 3 | 0 | 9 | 0.1166 | 0.1505 |
| 8 | 9 | 0.5758 | 0.5856 | 0.5499 | 3 | 3 | 0 | 9 | 0.1212 | 0.2012 |
| 9 | 10 | 0.6589 | 0.6755 | 0.6109 | 1 | 3 | 0 | 9 | 0.1554 | 0.6877 |
| 10 | 1 | 5.8165 | 6.074 | 0.6523 | 0 | 3 | 0 | 9 | 0.1262 | 0.1995 |
