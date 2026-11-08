# 21 RF post-hoc control checks

Post-hoc, not preregistered. Same 4,667 windows, window labels, nine held-out blocks, SMOTE and 300 trees; alert when score > 0.50.
rf_row_max_19 trains on row labels and scores a window by its highest row. Seed 42 on window statistics reproduced the saved predictions of experiment 12.

| target | model | aggregation | detected | fault_cycles | false_alarms | normal_cycles | cross_alerts | other_fault_cycles | macro_f1 | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System_Failure | rf_last_step_19 | seed 42 | 92 | 104 | 21 | 127 |  |  | 0.7132 | 0.5889 | 0.7673 |
| System_Failure | rf_row_max_19 | seed 42 | 98 | 104 | 11 | 127 |  |  | 0.8317 | 0.8053 | 0.8935 |
| System_Failure | rf_19_raw_window | seed 42 | 104 | 104 | 5 | 127 |  |  | 0.8482 | 0.8515 | 0.9288 |
| System_Failure | rf_19_raw_window | seed 43 | 104 | 104 | 4 | 127 |  |  | 0.8444 | 0.8522 | 0.9291 |
| System_Failure | rf_19_raw_window | seed 44 | 104 | 104 | 4 | 127 |  |  | 0.8408 | 0.8483 | 0.9304 |
| ProtectiveStop | rf_last_step_19 | seed 42 | 70 | 76 | 15 | 127 | 4 | 28 | 0.7939 | 0.656 | 0.9126 |
| ProtectiveStop | rf_row_max_19 | seed 42 | 69 | 76 | 1 | 127 | 2 | 28 | 0.8856 | 0.838 | 0.9697 |
| ProtectiveStop | rf_19_raw_window | seed 42 | 74 | 76 | 2 | 127 | 0 | 28 | 0.8871 | 0.8706 | 0.9651 |
| ProtectiveStop | rf_19_raw_window | seed 43 | 74 | 76 | 3 | 127 | 1 | 28 | 0.8847 | 0.8672 | 0.9645 |
| ProtectiveStop | rf_19_raw_window | seed 44 | 74 | 76 | 2 | 127 | 1 | 28 | 0.8834 | 0.8653 | 0.9627 |
| GripLost | rf_last_step_19 | seed 42 | 32 | 48 | 24 | 127 | 12 | 56 | 0.6124 | 0.3708 | 0.6929 |
| GripLost | rf_row_max_19 | seed 42 | 39 | 48 | 6 | 127 | 2 | 56 | 0.7699 | 0.6473 | 0.8616 |
| GripLost | rf_19_raw_window | seed 42 | 45 | 48 | 2 | 127 | 3 | 56 | 0.8159 | 0.7742 | 0.9267 |
| GripLost | rf_19_raw_window | seed 43 | 45 | 48 | 1 | 127 | 4 | 56 | 0.8088 | 0.78 | 0.9303 |
| GripLost | rf_19_raw_window | seed 44 | 45 | 48 | 1 | 127 | 3 | 56 | 0.8096 | 0.7746 | 0.9274 |
| System_Failure | rf_19_raw_window | 2 of 3 seeds | 104 | 104 | 4 | 127 |  |  |  |  |  |
| ProtectiveStop | rf_19_raw_window | 2 of 3 seeds | 74 | 76 | 2 | 127 | 1 | 28 |  |  |  |
| GripLost | rf_19_raw_window | 2 of 3 seeds | 45 | 48 | 1 | 127 | 3 | 56 |  |  |  |
