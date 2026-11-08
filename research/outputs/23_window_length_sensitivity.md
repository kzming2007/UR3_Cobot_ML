# 23 Window length sensitivity

Post-hoc, not preregistered. Same 19 sensors, nine held-out blocks and rules as experiment 12; only the window length changes. 10-step rows are the saved results of experiment 12.
Check: one 10-step CNN fold rerun through this path differs from experiment 12 by at most 0.00e+00 in score, same decisions: True.

| window_size | target | model | windows | positive_rate | cycles | detected | fault_cycles | false_alarms | normal_cycles | cross_alerts | other_fault_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | System_Failure | rf_19_raw_window | 5822 | 0.1773 | 231 | 103 | 104 | 5 | 127 |  |  |
| 10 | System_Failure | rf_19_raw_window | 4667 | 0.2492 | 231 | 104 | 104 | 5 | 127 |  |  |
| 20 | System_Failure | rf_19_raw_window | 2365 | 0.4465 | 230 | 102 | 104 | 9 | 126 |  |  |
| 5 | System_Failure | 1d_cnn_19_raw | 5822 | 0.1773 | 231 | 103 | 104 | 56 | 127 |  |  |
| 10 | System_Failure | 1d_cnn_19_raw | 4667 | 0.2492 | 231 | 104 | 104 | 23 | 127 |  |  |
| 20 | System_Failure | 1d_cnn_19_raw | 2365 | 0.4465 | 230 | 82 | 104 | 7 | 126 |  |  |
| 5 | ProtectiveStop | rf_19_raw_window | 5822 | 0.0984 | 231 | 72 | 76 | 7 | 127 | 0 | 28 |
| 10 | ProtectiveStop | rf_19_raw_window | 4667 | 0.1399 | 231 | 74 | 76 | 2 | 127 | 0 | 28 |
| 20 | ProtectiveStop | rf_19_raw_window | 2365 | 0.3006 | 230 | 70 | 76 | 5 | 126 | 1 | 28 |
| 5 | ProtectiveStop | 1d_cnn_19_raw | 5822 | 0.0984 | 231 | 74 | 76 | 25 | 127 | 3 | 28 |
| 10 | ProtectiveStop | 1d_cnn_19_raw | 4667 | 0.1399 | 231 | 73 | 76 | 17 | 127 | 4 | 28 |
| 20 | ProtectiveStop | 1d_cnn_19_raw | 2365 | 0.3006 | 230 | 65 | 76 | 12 | 126 | 1 | 28 |
| 5 | GripLost | rf_19_raw_window | 5822 | 0.0809 | 231 | 46 | 48 | 2 | 127 | 1 | 56 |
| 10 | GripLost | rf_19_raw_window | 4667 | 0.1146 | 231 | 45 | 48 | 2 | 127 | 3 | 56 |
| 20 | GripLost | rf_19_raw_window | 2365 | 0.1759 | 230 | 43 | 48 | 1 | 126 | 11 | 56 |
| 5 | GripLost | 1d_cnn_19_raw | 5822 | 0.0809 | 231 | 48 | 48 | 88 | 127 | 44 | 56 |
| 10 | GripLost | 1d_cnn_19_raw | 4667 | 0.1146 | 231 | 48 | 48 | 27 | 127 | 22 | 56 |
| 20 | GripLost | 1d_cnn_19_raw | 2365 | 0.1759 | 230 | 46 | 48 | 6 | 126 | 16 | 56 |
