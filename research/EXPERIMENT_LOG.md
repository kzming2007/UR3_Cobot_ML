# UR3 CobotOps 실험 일지

> [!note] 관리 원칙
> 실제로 실행한 실험만 시간순으로 누적한다. 입력되지 않은 조건이나 결과를 추정해서 채우지 않는다.

## 실험 기록

### 2026-07-08 - 00 데이터 감사

- 질문: 공개 `ur3_cobotops.csv`가 연구용 분류 실험에 어떤 구조와 제한을 갖는가?
- 데이터 및 입력: `dataset/ur3_cobotops.csv`
- 실행 조건: `python -X utf8 research\00_data_audit.py`
- 방법: 컬럼, 결측치, target 분포, timestamp 간격, cycle 구조, feature group을 점검했다.
- 결과: 원본 7,409 rows, 주요 결측 제거 후 모델 학습 기준 7,355 rows, unique cycle 240개를 확인했다. `System_Failure`는 `ProtectiveStop OR GripLost`로 정의했다.
- 해석: 공개 CSV의 timestamp는 대부분 약 1초 간격이며, `workload`, `movement speed`, `gripping force`는 직접 컬럼이 아니다.
- 한계 또는 오류: 일부 timestamp 간격이 길어 균일한 고주파 센서 데이터로 가정하면 안 된다.
- 다음 실험: row-level baseline과 cycle-aware split을 비교한다.
- 관련 파일: `research/outputs/00_data_audit_report.md`

### 2026-07-08 - 01 Row-level baseline 재현

- 질문: 기존 프로젝트의 분류 기반 baseline을 재현하고, random split과 cycle-group split 차이를 확인할 수 있는가?
- 데이터 및 입력: 원본 센서 feature 19개, power feature 6개, `Abs_Current_Sum`, targets `System_Failure`, `ProtectiveStop`, `GripLost`
- 실행 조건: `python -X utf8 research\01_baseline_reproduction.py`
- 방법: `Random Forest`, `SMOTE + Random Forest`를 row 단위 `random_stratified` split과 `cycle_group` split에서 비교했다.
- 결과: `cycle_group + rf_smote` 기준 Macro F1은 `System_Failure` 0.7901, `ProtectiveStop` 0.8428, `GripLost` 0.7623이었다.
- 해석: 기존 baseline은 재현되며, 고장 유형별 결과가 달라 `System_Failure` 통합 타깃만으로는 해석이 부족하다.
- 한계 또는 오류: row-level 입력은 시간 순서와 cycle 내부 변화를 직접 반영하지 않는다.
- 다음 실험: cycle 경계를 넘지 않는 window feature baseline을 만든다.
- 관련 파일: `research/outputs/01_baseline_results.md`

### 2026-07-08 - 02 Cycle 내부 window feature baseline

- 질문: cycle 내부의 짧은 시계열 구간을 feature로 요약하면 row-level baseline보다 분류 성능이 좋아지는가?
- 데이터 및 입력: cycle 경계를 넘지 않는 5, 10, 20 step window
- 실행 조건: `python -X utf8 research\02_window_feature_baseline.py`
- 방법: 각 window에서 mean, std, min, max, range, first-last delta, simple slope를 만들고 `cycle_group` split에서 `Random Forest`, `SMOTE + Random Forest`를 비교했다.
- 결과: `rf_smote` 기준 best Macro F1은 `System_Failure` 10-step 0.8593, `ProtectiveStop` 10-step 0.9323, `GripLost` 5-step 0.8796이었다.
- 해석: 구간 단위 이상탐지에서는 window feature가 row-level baseline보다 세 target 모두에서 개선됐다.
- 한계 또는 오류: window 안에 positive가 하나라도 있으면 positive로 두므로, 이미 발생한 이상 상태를 포함할 수 있다. 고장 발생 전 예측으로 해석하지 않는다.
- 다음 실험: first positive 이후 데이터를 제외한 pre-failure window baseline을 만든다.
- 관련 파일: `research/outputs/02_window_feature_results.md`, `research/outputs/02_row_vs_window_comparison.csv`

### 2026-07-09 - 03 First positive 이전 pre-failure window baseline

- 질문: 이미 positive가 된 이후 데이터를 제외하고도 고장 발생 직전 구간을 분류할 수 있는가?
- 데이터 및 입력: 각 cycle의 first positive step 이전 window, prediction horizon 3, 5, 10 step
- 실행 조건: `python -X utf8 research\03_pre_failure_window_baseline.py`
- 방법: first positive step 이후 또는 그 시점을 포함하는 window를 제외했다. window 종료 시점이 first positive step 이전 horizon 안에 있으면 positive로 두고, `cycle_group` split에서 `Random Forest`, `SMOTE + Random Forest`를 비교했다.
- 결과: `rf_smote` best 기준 `GripLost`는 window 5, horizon 3에서 positive F1 0.4375, `ProtectiveStop`은 window 10, horizon 3에서 positive F1 0.5000, `System_Failure`는 window 5, horizon 10에서 positive F1 0.0727이었다.
- 해석: pre-failure task는 구간 단위 이상탐지보다 훨씬 어렵다. `GripLost`와 `ProtectiveStop`은 일부 조합에서 약한 사전 신호가 보이나, `System_Failure`는 default threshold에서 거의 잡히지 않았다.
- 한계 또는 오류: positive window 수가 적고 단일 split 결과라 안정적인 결론으로 보기 어렵다. ROC-AUC나 PR-AUC가 높더라도 positive recall이 0에 가까운 조합은 실제 경고 모델로 해석하지 않는다.
- 다음 실험: 반복 cycle-group split 또는 group cross-validation으로 안정성을 확인하고, threshold 조정을 별도 실험으로 분리한다.
- 관련 파일: `research/outputs/03_pre_failure_window_results.md`, `research/outputs/03_pre_failure_window_best.csv`

### 2026-07-22 - 04 Pre-failure 반복 cycle-group split 검증

- 질문: `03` pre-failure 대표 조합의 결과가 단일 split 우연이 아니라 반복 split에서도 유지되는가?
- 데이터 및 입력: `System_Failure` 5-step/horizon 10, `ProtectiveStop` 10-step/horizon 3, `GripLost` 5-step/horizon 3
- 실행 조건: `python -X utf8 research\04_pre_failure_repeated_split_validation.py`
- 방법: target별 유효 `cycle_group` split 30회를 만들고 `SMOTE + Random Forest`를 반복 학습했다. default threshold 기준 positive recall, precision, F1, `runs_with_tp_rate`를 집계했다.
- 결과: `GripLost`는 30회 모두 positive를 1개 이상 잡았고 positive F1 평균은 0.3164였다. `ProtectiveStop`은 positive F1 평균 0.0656, `runs_with_tp_rate` 0.1로 단일 split 결과가 안정적이지 않았다. `System_Failure`는 positive F1 평균 0.0963으로 낮았다.
- 해석: pre-failure default threshold 기준으로는 `GripLost`가 가장 안정적인 개별 타깃이다. `ProtectiveStop`은 test positive 수가 평균 5개로 작아 결과 변동이 크고, `System_Failure`는 통합 타깃이라 사전 예측 성능이 낮다.
- 한계 또는 오류: 반복 split은 대표 조합만 대상으로 했으며, 전체 window/horizon 조합을 모두 반복 검증한 것은 아니다.
- 다음 실험: threshold를 낮춰 recall을 회수할 수 있는지 확인한다.
- 관련 파일: `research/outputs/04_pre_failure_repeated_split_results.md`, `research/outputs/04_pre_failure_repeated_split_summary.csv`

### 2026-07-22 - 05 Pre-failure threshold 민감도 검증

- 질문: default threshold 0.50에서 놓친 pre-failure positive를 threshold 조정으로 회수할 수 있는가?
- 데이터 및 입력: `04`와 같은 target별 대표 조합, threshold 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
- 실행 조건: `python -X utf8 research\05_pre_failure_threshold_sensitivity.py`
- 방법: target별 유효 `cycle_group` split 30회에서 `SMOTE + Random Forest`를 학습하고, 같은 score에 여러 threshold를 적용했다. 평균 false positive rate가 0.10 이하인 후보 중 positive F1 평균이 가장 높은 threshold를 추천 후보로 정했다.
- 결과: 추천 후보는 `GripLost` threshold 0.30, `ProtectiveStop` threshold 0.15, `System_Failure` threshold 0.30이었다. `GripLost`는 threshold 0.30에서 positive recall 평균 0.4284, precision 평균 0.3723, positive F1 평균 0.3803, false positive rate 평균 0.0125였다. `System_Failure`는 threshold 0.30에서 recall은 0.2669로 올라갔지만 positive F1 평균은 0.2069에 그쳤다.
- 해석: threshold 조정은 `GripLost`에서 가장 해석 가능하며, `System_Failure`는 threshold를 낮춰도 통합 타깃의 어려움이 남는다. `ProtectiveStop`은 false positive rate는 낮지만 positive 수가 작아 안정적인 주장으로 쓰기 어렵다.
- 한계 또는 오류: 추천 threshold는 후속 검증 후보일 뿐이며, 최종 결론에는 false positive 부담과 반복 split 분산을 함께 보고해야 한다.
- 다음 실험: 딥러닝 비교 전, `GripLost` 중심 pre-failure 설정을 유지할지 결정한다.
- 관련 파일: `research/outputs/05_pre_failure_threshold_sensitivity_results.md`, `research/outputs/05_pre_failure_threshold_sensitivity_recommended.csv`

### 2026-08-10 - 06 공정조건 분리 가능성 진단

- 질문: 논문에 제시된 `movement speed`, `workload`, `gripping force`의 서로 다른 수준을 공개 센서 데이터만으로 cycle별 독립 구분할 수 있는가?
- 데이터 및 입력: `dataset/ur3_cobotops.csv`, `dataset/dataset_02052023.xlsx`, IF-FCM 논문 DOI `10.1007/978-3-031-63851-0_6`
- 실행 조건: `python -X utf8 research\06_process_condition_separability.py`
- 방법: 논문의 25회 반복 기술을 근거로 25-cycle 후보 블록을 구성했다. 비연속적으로 재등장한 동일 `cycle` ID는 시간순 연속 구간별 `cycle_run`으로 분리했다. first positive 이전 healthy prefix에서 speed/current/Tool current 요약을 만들고, 조건별 K-Means 3-cluster, GMM BIC 1-5 component, 후보 블록 분류를 비교했다.
- 결과: `movement_speed` Silhouette 0.2044, `workload` 0.4037, `gripping_force` 0.2728이었고 세 조건 모두 GMM BIC가 시험 범위에서 5 components를 선택했다. process sensor Logistic Regression의 9개 후보 블록 balanced accuracy는 0.2060(chance 0.1111), temperature-only Random Forest는 0.9620이었다.
- 해석: 센서에 일부 regime 차이는 있으나 논문의 세 수준과 일치하는 단순한 3개 구조는 확인되지 않았다. 후보 블록 정체성은 공정 sensor보다 온도와 수집 순서의 영향을 크게 받아 thermal/session drift 교란 가능성이 높다.
- 한계 또는 오류: 공개 Excel/CSV와 검색 가능한 부속자료에는 cycle-to-condition 정답표가 없다. 따라서 cluster를 실제 1/2/3 kg, 60/80/100%, 80/100/120 N으로 이름 붙이거나 supervised accuracy를 계산할 수 없다. healthy prefix 5 rows 미만 시행 71개를 제외해 선택 편향 가능성도 있다.
- 추가 발견: raw `cycle` ID 224, 225, 226, 227, 229, 230, 231, 232, 233이 서로 떨어진 시행 구간에 재등장한다. 기존 `02`-`05`가 raw `cycle`만으로 window/group을 만들었으므로 `cycle_run` 기준 재검증이 필요하다.
- 다음 실험: 기존 window/pre-failure 실험을 `cycle_run` 기준으로 다시 실행해 차이를 확인한다. 실제 공정조건 분류는 저자 또는 원 수집팀의 cycle-to-condition mapping 확보 후 진행한다.
- 관련 파일: `research/outputs/06_process_condition_separability.md`, `research/outputs/06_condition_cluster_summary.csv`, `research/outputs/06_condition_block_classification.csv`

### 2026-08-10 - 07 Cycle-run 경계 재검증

- 질문: 비연속적으로 재등장한 raw `cycle` ID를 분리하면 기존 `02`-`05` 결론이 유지되는가?
- 데이터 및 입력: 수정 전 요약 결과 `research/outputs/raw_cycle_reference/`, 수정 후 `cycle_run` 기준 `02`-`05` 결과
- 실행 조건: `python -X utf8 research\02_window_feature_baseline.py`, `03_pre_failure_window_baseline.py`, `04_pre_failure_repeated_split_validation.py`, `05_pre_failure_threshold_sensitivity.py`, `07_cycle_run_revalidation.py`
- 방법: 행 순서에서 `cycle` 값이 바뀔 때마다 `cycle_run`을 부여하고, window 생성·first positive 탐색·group split을 `cycle_run` 기준으로 다시 수행했다.
- 결과: 5/10/20-step에서 target별로 각각 24/54/104개의 경계-crossing window가 제거됐다. `02`와 `03`의 단일 split best 설정은 일부 바뀌었다. 동일 설정 30회 반복 positive F1 평균은 `GripLost` 0.3164 -> 0.3058, `ProtectiveStop` 0.0656 -> 0.0867, `System_Failure` 0.0963 -> 0.0920이었다.
- 해석: raw `cycle` 경계 문제는 수정이 필요했고 단일 best 수치는 불안정했다. 그러나 `GripLost`가 상대적으로 가장 안정적인 pre-failure 타깃이라는 반복 검증 결론은 유지됐다.
- threshold 결과: `GripLost`와 `System_Failure` 추천 후보는 0.30을 유지했고, `ProtectiveStop`은 0.15에서 0.20으로 바뀌었다. 수정 후 positive F1 평균은 각각 0.4101, 0.1856, 0.3231이었다.
- 한계 또는 오류: 수정 전후 단일 split은 test 시행 구성이 달라져 delta를 순수한 경계 효과로만 해석할 수 없다. 반복 split 비교를 주 근거로 사용한다.
- 다음 실험: acquisition block 전체를 제외하는 구간 단위 이상탐지 강건성 평가를 수행한다.
- 관련 파일: `research/outputs/07_cycle_run_revalidation.md`, `research/outputs/07_cycle_run_04_repeated_comparison.csv`, `research/outputs/07_cycle_run_05_threshold_comparison.csv`

### 2026-08-10 - 08 Acquisition block held-out 강건성 검증

- 질문: 특정 25-cycle acquisition block 전체를 학습에서 제외해도 구간 단위 이상탐지가 일반화되는가?
- 데이터 및 입력: `cycle_run` 경계의 5/10/20-step window, 20개 이상 cycle_run이 있는 후보 block 1, 2, 3, 4, 5, 7, 8, 9, 10
- 실행 조건: `python -X utf8 research\08_block_held_out_robustness.py`
- 방법: 한 block 전체를 test로 두고 나머지 8개 block으로 `SMOTE + Random Forest`를 학습했다. 전체 sensor와 temperature 제거 feature를 비교하고 9개 holdout의 평균·표준편차·최솟값을 집계했다.
- 결과: 모든 target/window/feature 설정에서 9개 block 모두 positive를 1개 이상 탐지했다. 대표 결과는 `System_Failure` 10-step/no-temperature Macro F1 0.8580, positive recall 0.7794, positive F1 0.7839, 최소 recall 0.4872였다. `ProtectiveStop` 10-step/all-sensors는 0.8650/0.7594/0.7620/0.2593, `GripLost` 20-step/no-temperature는 0.8759/0.7337/0.7911/0.3415였다.
- 해석: 구간 단위 이상탐지는 특정 acquisition block 하나에만 의존한 결과가 아니며, 공정조건 정답표 없이도 block/session 변화에 대한 일반화 성능을 평가할 수 있다.
- 한계 또는 오류: 후보 block은 실제 공정조건 라벨이 아니라 condition과 time/session drift가 섞인 proxy다. window 안에 이미 이상 상태가 포함될 수 있으므로 pre-failure 예측 결과로 해석하지 않는다.
- temperature ablation: `System_Failure`에서는 제거 효과가 비교적 일관됐지만 다른 target에서는 metric별 효과가 엇갈려, 온도를 일괄 제외하기보다 ablation으로 보고한다.
- 다음 실험: 동일 `cycle_run` 경계와 block-held-out 분할을 고정해 window feature와 sequence model을 비교한다.
- 관련 파일: `research/outputs/08_block_held_out_robustness.md`, `research/outputs/08_block_held_out_summary.csv`, `research/outputs/08_block_held_out_best.csv`

### 2026-08-12 - 09 공통 10-step 사건 단위 block-held-out 검증

- 질문: 딥러닝 비교 전에 10-step과 동일 후보 block 분할을 고정했을 때, Random Forest가 실제 이상 event cycle을 얼마나 탐지하고 정상 cycle에서 얼마나 오경보를 내는가?
- 데이터 및 입력: `cycle_run` 경계의 10-step window, 후보 block 1, 2, 3, 4, 5, 7, 8, 9, 10, 전체 sensor 및 temperature 제거 feature.
- 실행 조건: `python -X utf8 research\09_event_level_block_validation.py`, `SMOTE + Random Forest`, 300 trees, positive decision rule `score > 0.50`.
- 방법: 각 block을 한 번씩 test로 두고 나머지 8개 block만 학습했다. 실제 positive가 포함된 window를 하나 이상 맞힌 event cycle을 탐지로 계산하고, 정상 cycle에서 positive 예측이 하나라도 발생하면 cycle 오경보로 계산했다. 모든 window score와 cycle별 집계 결과를 저장했다.
- 결과: 전체 sensor 기준 `System_Failure`는 event cycle 87/87 탐지, 정상 cycle 오경보율 0.0522였다. `ProtectiveStop`은 59/62 탐지와 오경보율 0.0429, `GripLost`는 37/39 탐지와 오경보율 0.0429였다. 가장 어려운 block의 event recall은 각각 1.0, 0.5, 0.6667이었다.
- Temperature ablation: temperature 제거 시 `System_Failure` window positive F1은 0.7451에서 0.7839로 증가했지만 정상 cycle 오경보율도 0.0522에서 0.0783으로 증가했다. `ProtectiveStop`과 `GripLost`에서도 오경보율이 증가해 전체 sensor를 주 설정으로 유지한다.
- 정합성 검증: 54개 block 결과의 window-level Macro F1, positive recall, positive F1, PR-AUC가 기존 `08`의 동일 10-step 결과와 모두 일치했다. 저장된 24,210개 window 예측에서 summary를 재집계한 결과도 원본 summary와 일치했다.
- 해석: 구간 단위 이상탐지는 사건 단위에서도 높은 탐지율을 보이지만 `ProtectiveStop`과 `GripLost`는 특정 block에서 누락이 남는다. 이 결과는 이상이 포함된 window 탐지이며 고장 발생 전 예측으로 해석하지 않는다.
- 한계 또는 오류: 첫 실행은 120초 제한으로 종료되어 결과가 저장되지 않았다. 동일한 300-tree 설정으로 실행 시간을 늘려 134.6초에 완료했으며, 0.5 동률 처리 차이를 발견해 기존 `RandomForest.predict()`와 같도록 `score > 0.50`으로 저장 score를 재집계했다.
- 다음 실험: 전체 sensor, 10-step, 동일 9개 block, 동일 사건 단위 지표를 고정해 1D CNN과 LSTM을 비교한다.
- 관련 파일: `research/outputs/09_event_level_block_validation.md`, `research/outputs/09_event_level_block_summary.csv`, `research/outputs/09_event_level_cycle_results.csv`, `research/outputs/09_event_level_window_predictions.csv`

### 2026-08-12 - 10 고정 1D CNN/LSTM sequence model 비교

- 질문: 동일한 10-step 구간에서 step 순서를 직접 입력받는 소형 1D CNN과 LSTM이 통계적 window feature 기반 Random Forest보다 이상 event를 안정적으로 탐지하는가?
- 데이터 및 입력: `cycle_run` 경계를 넘지 않는 10-step×26-feature raw sequence, target `System_Failure`, `ProtectiveStop`, `GripLost`, 후보 block 1, 2, 3, 4, 5, 7, 8, 9, 10.
- 실행 조건: 데이터 준비와 집계는 프로젝트 Python 환경을 사용했다. 학습은 기존 `D:\Projects\Cube_Codex\.venvs\sam2_clean` 환경의 Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Ti를 패키지 변경 없이 재사용했다.
- 방법: 각 block을 한 번씩 test로 두고, 다음 순서의 outer-train block 1개를 validation으로 사용해 epoch를 선택했다. 선택된 epoch로 test block을 제외한 8개 block 전체를 재학습했다. 고정 소형 1D CNN과 단층 LSTM에 class-weighted BCE, Adam, threshold `score > 0.50`, seed 42/43/44를 적용했다.
- 실행 범위: 3 targets×2 models×9 test blocks×3 seeds의 162회 학습과 72,630개 window prediction을 완료했다. 각 target/model/seed 조합은 9개 test block과 4,035개 window를 모두 포함했고, test/validation block 분리와 threshold 규칙을 검증했다. 기록된 개별 학습 시간의 합은 486.7초였다.
- 결과: `System_Failure`의 event recall/정상 cycle 오경보율은 1D CNN 0.9732/0.1594, LSTM 0.9617/0.2058, Random Forest 1.0000/0.0522였다. `ProtectiveStop`은 각각 0.9355/0.2024, 0.9086/0.2119, 0.9516/0.0429였고, `GripLost`는 0.9487/0.3252, 0.9487/0.3763, 0.9487/0.0429였다.
- Window 결과: pooled positive F1 기준 `System_Failure`는 1D CNN 0.7097, LSTM 0.6749, Random Forest 0.7326이었다. `ProtectiveStop`은 0.6769, 0.6953, 0.7861이었고, `GripLost`는 0.5451, 0.5302, 0.6328이었다. PR-AUC도 세 target 모두 Random Forest가 가장 높았다.
- 해석: 소형 sequence model도 event cycle 대부분을 탐지했지만 정상 cycle에서 경보가 누적됐다. 현재 데이터 규모와 고정 block 평가에서는 짧은 시계열 구간을 통계량으로 정제한 Random Forest가 더 균형 잡힌 기준선이다. 딥러닝 사용 자체가 성능 향상을 보장하지 않는다는 비교 결과로 해석한다.
- 한계 또는 오류: window 안에 이미 positive 상태가 포함될 수 있어 pre-failure 예측 결과가 아니다. 후보 block은 실제 공정조건 라벨이 아니라 cycle 번호 기반 proxy다. Random Forest는 고정 1회, deep learning은 3개 seed 평균이며, architecture 탐색은 수행하지 않았다.
- 다음 실험: 추가 모델을 바로 늘리지 않고 false alarm이 집중된 정상 cycle과 성능이 낮은 held-out block을 먼저 분석할지 결정한다.
- 관련 파일: `research/10_prepare_sequence_data.py`, `research/10_torch_sequence_models.py`, `research/10_sequence_model_results.py`, `research/outputs/10_sequence_model_comparison.md`, `research/outputs/10_sequence_model_summary.csv`, `research/outputs/10_sequence_training_runs.csv`

### 2026-08-23 - 11 Sequence model 오류 반복성 및 block 집중도 분석

- 질문: `10`에서 증가한 딥러닝 오경보가 seed별 우연인지, 특정 cycle과 held-out block에서 반복되는 구조적 오류인지 확인할 수 있는가?
- 데이터 및 입력: `09`의 all-sensors Random Forest prediction, `10`의 1D CNN/LSTM 3-seed prediction, `10`의 10-step×26-feature sequence cache. 새 모델 학습은 수행하지 않았다.
- 실행 조건: `python -X utf8 research\11_sequence_error_analysis.py`.
- 방법: 딥러닝은 3개 seed 중 2개 이상이 같은 cycle에서 경보하면 cycle consensus 오경보로 정의했다. Event detection도 2개 이상 seed가 실제 positive window를 하나 이상 탐지해야 consensus 탐지로 두었다. Random Forest는 고정 1회 결과를 사용했다. 정상 cycle 내부에서 동일 window에 2/3 seed consensus가 발생한 구간과 true-negative 구간의 feature mean/range 중앙값 차이를 true-negative IQR로 표준화해 기술통계로 비교했다.
- Consensus 결과: 정상 cycle 오경보율은 `System_Failure`에서 Random Forest 0.0522, 1D CNN 0.1391, LSTM 0.1739였다. `ProtectiveStop`은 0.0429/0.1857/0.2143, `GripLost`는 0.0429/0.3067/0.3558이었다. Consensus event recall은 각각 `System_Failure` 1.0000/0.9655/0.9885, `ProtectiveStop` 0.9516/0.9355/0.9032, `GripLost` 0.9487/0.9487/0.9744였다.
- Seed 반복성: 3개 seed 모두에서 경보한 정상 cycle은 `System_Failure` 1D CNN/LSTM 10/9개, `ProtectiveStop` 15/20개, `GripLost` 32/35개였다. 오경보 증가를 초기화 seed의 일시적 변동만으로 설명하기 어렵다.
- 모델 간 오류 겹침: Random Forest는 정상으로 처리했지만 deep learning만 consensus 오경보를 낸 cycle은 `System_Failure` 1D CNN/LSTM 15/19개, `ProtectiveStop` 23/26개, `GripLost` 47/54개였다. 딥러닝 오류 대부분은 Random Forest 오류의 단순 재현이 아니었다.
- Block 집중: 가장 높은 consensus 오경보율은 `GripLost` LSTM block 8의 0.8000, `GripLost` 1D CNN block 10의 0.7619, `System_Failure` LSTM block 10의 0.7143이었다. `ProtectiveStop`은 두 모델 모두 block 10에서 0.5000이었고 1D CNN은 block 5에서도 0.4500이었다.
- Sensor 기술통계: 정상 cycle의 반복 오경보 window에서 `System_Failure` 1D CNN은 `Current_J3` 평균과 `Temperature_J0/J1` 평균, LSTM은 `Tool_current` mean/range 차이가 상대적으로 컸다. `ProtectiveStop`은 `Current_J2/J3` 평균과 `Speed_J4` range, `GripLost`는 `Abs_Current_Sum` mean/range 차이가 반복됐다.
- 해석: 딥러닝 오경보는 seed가 바뀌어도 같은 cycle에서 반복되고 일부 block에 집중됐다. 이는 현재 sequence model이 짧은 구간의 특정 저변동·센서 수준 패턴과 수집 구간 변화를 이상으로 오인했을 가능성을 제시하지만, sensor shift는 겹치는 window의 사후 기술통계이므로 원인 또는 feature importance로 확정하지 않는다.
- 다음 판단: 모델 규모 확대는 보류한다. 공동연구자와 구간 단위 이상탐지를 주 결과로 확정할지, 제출 계획서의 미수행 항목인 정상-only LSTM Autoencoder를 동일 평가 규칙으로 1개만 추가할지 합의한다.
- 관련 파일: `research/11_sequence_error_analysis.py`, `research/outputs/11_sequence_error_analysis.md`, `research/outputs/11_error_cycle_details.csv`, `research/outputs/11_error_pairwise_summary.csv`, `research/outputs/11_false_alarm_sensor_shifts.csv`

### 2026-08-23 - 12 동일 19개 센서 기반 LSTM Autoencoder 후속 비교

- 질문: 동일한 센서 입력과 block-held-out 평가에서 supervised 모델과 정상-only LSTM Autoencoder의 차이는 무엇이며, Autoencoder가 정상 cycle 오경보를 줄이면서 이상 event를 탐지하는가?
- 사전 고정: 전체 결과 실행 전에 `research/2026-08-23_lstm_autoencoder_preregistration.md`를 커밋했다. 주 연구는 supervised 구간 단위 이상탐지로 두고, Autoencoder는 탐색적 부가 모델로 정했다. Primary threshold는 calibration 정상 cycle 최대 reconstruction error의 q95로 고정했으며 q90과 q97.5는 민감도 분석으로만 사용했다.
- 공통 입력: `cycle_run` 경계를 넘지 않는 10-step×19개 원본 센서와 202개 cycle을 사용했다. Event/normal cycle은 `System_Failure` 87/115개, `ProtectiveStop` 62/140개, `GripLost` 39/163개다.
- Random Forest: 각 센서에서 mean, std, min, max, range, delta, slope를 계산한 133개 window feature에 SMOTE와 300-tree Random Forest를 적용했다. 9개 test block을 한 번씩 평가했고 threshold는 `score > 0.50`이다.
- 1D CNN: 같은 10×19 sequence를 직접 입력했다. 각 outer fold의 train 8개 block 중 7개로 epoch를 선택한 뒤 8개 전체로 재학습했다. Class-weighted BCE, Adam, threshold `score > 0.50`, seed 42/43/44를 사용했다.
- LSTM Autoencoder: `System_Failure=0`인 완전 정상 cycle만 학습했다. Outer fold마다 test 1개, validation 1개, calibration 2개, core train 5개 block으로 분리했다. Epoch 선택 후 core+validation 6개 block으로 재학습했으며, calibration block은 학습에 사용하지 않았다. 단층 hidden size 32, MSE, Adam learning rate 0.001, 최대 40 epochs, patience 5, seed 42/43/44를 사용했다.
- 실행 범위: Random Forest 27회, 1D CNN 81회, LSTM Autoencoder 27회를 실행했다. Window prediction은 각각 12,105개와 145,260개이며, Autoencoder calibration 정상 cycle 수는 fold별 13~36개였다. PyTorch 학습 기록 시간 합은 395.0초다.
- Primary 결과: Random Forest의 event cycle recall/정상 cycle 오경보율은 `System_Failure` 0.9770/0.0435, `ProtectiveStop` 0.9516/0.0286, `GripLost` 0.9231/0.0368이었다. 1D CNN의 2/3 seed consensus는 각각 0.9770/0.1913, 0.9355/0.1929, 0.9231/0.2454였다. LSTM Autoencoder q95는 각각 0.0460/0.1304, 0.0645/0.1071, 0.0000/0.1227이었다.
- Threshold 민감도: q90으로 낮춰도 Autoencoder event recall은 `System_Failure` 0.0805, `ProtectiveStop` 0.1129, `GripLost` 0.0000에 그쳤고 오경보율은 각각 0.2087, 0.1714, 0.1963으로 증가했다. q97.5도 q95의 결론을 바꾸지 않았다.
- 해석: 현재 데이터에서 정상 reconstruction error가 고장 유형을 공통으로 분리한다는 가정은 지지되지 않았다. 특히 `GripLost` event는 q90에서도 탐지하지 못했다. Random Forest가 세 타깃 모두에서 recall과 오경보 방향으로 우세했으므로, Autoencoder를 성능 향상 근거로 사용하지 않는다. 이 결과는 기본 모델의 강한 기준선과 딥러닝 가정의 실패 조건을 비교했다는 점에서 연구 결과로 보고한다.
- 한계: 겹치는 window 4,035개를 독립 표본으로 해석하지 않는다. Cycle-level n은 202개이며, block/session 상관이 남는다. 9개 block은 공정조건 정답이 아니라 수집 구간 proxy이고, 동일 데이터에 대한 내부 후속 비교라 외부 검증이 아니다. Calibration 정상 cycle 수가 적어 quantile 임계값의 해상도도 제한된다.
- 다음 판단: 새 architecture나 threshold 탐색을 추가하지 않는다. 최종 보고서는 supervised 구간 탐지를 주 결과로 두고, Autoencoder는 사전 고정한 부정적 비교 결과, pre-failure는 `GripLost` 중심의 제한적 부가 분석으로 구성할지 공동연구자와 합의한다.
- 관련 파일: `research/2026-08-23_lstm_autoencoder_preregistration.md`, `research/12_matched_rf_baseline.py`, `research/12_matched_torch_models.py`, `research/12_matched_results.py`, `research/outputs/12_matched_lstm_autoencoder_comparison.md`, `research/outputs/12_matched_consensus_summary.csv`, `research/outputs/12_matched_pairwise_cycle_errors.csv`

### 2026-08-23 - 13 최종 평가표 재집계

- 질문: 저장된 `12` 예측을 최종 보고에 쓸 confusion matrix와 ROC-AUC·PR-AUC 표로 정리하면 어떤 값이 되는가?
- 방법: `12`의 window 예측을 재집계했다. 새 모델을 학습하지 않았다.
- 관련 파일: `research/13_final_evaluation_tables.py`, `research/outputs/13_final_evaluation_tables.md`, `research/outputs/13_window_model_summary.csv`

### 2026-08-23 - 14 센서 그룹 ablation

- 질문: 기존 프로젝트에서 반복적으로 중요하게 해석한 전류 계열이 동일 19-sensor·9-block 평가에서도 실제 예측에 기여하는가? 전류·속도·온도·`Tool_current`의 기여는 타깃별로 같은가?
- 사전 고정: 전체 실행 전에 `research/2026-08-23_sensor_group_ablation_preregistration.md`를 별도 커밋했다. `all_19`를 공통 기준선으로 두고, 센서 그룹 단독 입력 4개와 제거 입력 5개를 고정했다. 결과를 본 뒤 입력 조합이나 threshold를 바꾸지 않기로 했다.
- 공통 조건: `cycle_run` 경계를 넘지 않는 10-step window, 19개 원본 센서의 mean·std·min·max·range·delta·slope, 9개 acquisition block held-out, SMOTE와 300-tree Random Forest, `random_state=42`, threshold `score > 0.50`을 사용했다.
- 실행 범위: 3 targets×10 sensor variants×9 test blocks의 270회 학습을 완료했다. 121,050개 window prediction, 6,060개 cycle 결과와 54개 paired error 요약을 저장했다. 개별 실행 시간 합은 635.5초였다.
- 정합성 검증: `all_19`의 27개 fold score와 prediction이 `12`의 Random Forest 결과와 정확히 일치했다. Variant 간 window 메타데이터와 라벨, threshold 재구성, feature 수, 202개 cycle 포함 여부를 확인했다.
- 전류 계열 제거: `System_Failure`의 event recall/정상 cycle 오경보율은 `0.9770/0.0435`에서 `0.9310/0.1217`로 악화됐다. `GripLost`는 `0.9231/0.0368`에서 `0.4103/0.0368`로 recall이 크게 낮아졌다. `ProtectiveStop`은 `0.9516/0.0286`에서 `0.9677/0.0786`으로 recall과 오경보가 함께 증가했다.
- `Tool_current` 제거: `System_Failure`와 `ProtectiveStop`은 recall이 각각 `1.0000`, `0.9677`로 높아지고 오경보율은 기준선과 같았다. `GripLost`는 recall `0.9231`이 유지됐지만 오경보율이 `0.0429`로 소폭 증가했다.
- 단독 입력: 관절 전류만 사용하면 recall은 `0.9516`~`0.9885`였지만 오경보율이 `0.0643`~`0.1217`로 높았다. 속도만 사용해도 recall은 `0.9231`~`0.9839`였지만 오경보율이 `0.1214`~`0.3006`이었다. 온도만 사용한 recall은 `0.0256`~`0.2529`로 낮았다.
- 해석: 관절 전류 계열은 특히 `GripLost`와 `System_Failure` 탐지에 기여했지만, 모든 전류 센서가 항상 유용하다는 결론은 성립하지 않았다. 속도는 단독으로도 event를 잡지만 정상 cycle 오경보를 억제하지 못했고, 온도는 단독 탐지력보다 수집 구간 차이 보정에 기여했을 가능성이 있다.
- 한계: 센서 그룹 변경은 feature 차원과 SMOTE 공간도 바꾸므로 인과적 센서 중요도를 증명하지 않는다. Random Forest seed는 42로 고정했고, 동일 데이터 내부 비교이므로 외부 일반화 근거가 아니다. 일부 window 지표가 개선돼도 cycle 오경보가 악화돼 cycle 지표를 우선 해석한다.
- 다음 판단: `all_19`를 최종 공통 기준선으로 유지한다. 이번 결과만으로 `drop_tool_current`를 새 주 모델로 선택하거나 추가 조합을 탐색하지 않는다. 동일 조건 Logistic Regression·SVM이 계획서 이행에 필요한지는 공동연구자와 별도로 결정한다.
- 관련 파일: `research/2026-08-23_sensor_group_ablation_preregistration.md`, `research/14_sensor_group_ablation.py`, `research/14_sensor_group_ablation_results.py`, `research/outputs/14_sensor_group_ablation.md`, `research/outputs/14_sensor_group_ablation_summary.csv`, `research/outputs/14_sensor_group_ablation_paired_errors.csv`

### 2026-08-23 - 15 동일 시계열 특징 기반 기본 분류 모델 비교

- 질문: 동일한 10-step×19-sensor 시계열 통계 특징에서 Logistic Regression, RBF SVM, Random Forest를 비교하면 특징 정제의 효과와 모델 구조의 효과를 구분할 수 있는가?
- 사전 고정: 전체 실행 전에 `research/2026-08-23_classical_model_comparison_preregistration.md`를 별도 커밋했다. Logistic Regression L2 `C=1`, RBF SVM `C=1`, `gamma=scale`, 공통 threshold `score > 0.50`을 고정하고 tuning을 수행하지 않았다.
- 공통 조건: `cycle_run` 경계를 넘지 않는 10-step window, 19개 원본 센서의 133개 통계 특징, 9개 acquisition block held-out, 학습 fold 내부 SMOTE를 사용했다. Logistic Regression과 SVM에는 학습 fold 내부 `StandardScaler`를 적용했다. Random Forest는 `12`의 고정 prediction을 재사용했다.
- 실행 범위: 3 targets×2개 새 모델×9 test blocks의 54회 학습과 24,210개 새 window prediction을 완료했다. 기존 Random Forest를 합쳐 모델별 4,035개 window와 202개 cycle을 비교했다. 기록된 새 학습 시간 합은 779.1초였다.
- 실행 검증: 모든 fold의 test window 메타데이터와 label이 Random Forest 기준선과 일치했다. 비유한 score와 fit 실패는 없었고 Logistic Regression 27개 fold는 모두 `max_iter=5000` 전에 수렴했다. 최대 반복 횟수는 4,757회였다.
- `System_Failure`: Random Forest, Logistic Regression, RBF SVM의 event recall/정상 cycle 오경보율은 각각 `0.9770/0.0435`, `0.9885/0.5304`, `0.9540/0.0957`이었다.
- `ProtectiveStop`: 세 모델은 각각 `0.9516/0.0286`, `0.9677/0.2714`, `0.9032/0.0214`였다. SVM은 오경보 cycle이 Random Forest보다 1개 적었지만 event cycle을 3개 더 놓쳤다.
- `GripLost`: 세 모델은 각각 `0.9231/0.0368`, `1.0000/0.4908`, `0.9487/0.1779`였다.
- Paired 오류: Logistic Regression은 Random Forest가 정상으로 처리한 cycle에서 새 오경보를 `System_Failure` 58개, `ProtectiveStop` 34개, `GripLost` 75개 만들었다. RBF SVM의 새 오경보는 각각 8개, 1개, 25개였다.
- 해석: 시계열 통계 특징에는 선형 모델도 높은 event recall을 낼 수 있는 신호가 남지만, Logistic Regression은 정상 cycle을 충분히 배제하지 못했다. RBF SVM은 이를 줄였지만 타깃별 trade-off가 남았다. 현재 고정 조건에서는 Random Forest가 탐지율과 오경보율을 가장 안정적으로 균형화했다.
- 한계: 대표 기본 설정 한 개씩만 비교했으므로 각 모델 계열의 최적 성능이 아니다. SVM과 Random Forest의 probability 산출 방식은 다르다. 동일 데이터 내부 비교이며 조기 고장 예측이나 외부 일반화 결과가 아니다.
- 다음 판단: 결과를 본 뒤 Grid Search, kernel, `C`, `gamma`, threshold를 추가로 조정하지 않는다. 기본 모델 비교를 최종 보고서 본문의 통제 실험으로 둘지 부록으로 둘지만 공동연구자와 결정한다.
- 관련 파일: `research/2026-08-23_classical_model_comparison_preregistration.md`, `research/15_classical_model_comparison.py`, `research/15_classical_model_comparison_results.py`, `research/outputs/15_classical_model_comparison.md`, `research/outputs/15_classical_model_summary.csv`, `research/outputs/15_classical_model_paired_errors.csv`

### 2026-08-26 - 16 완전 정상 오경보와 교차 경보 분리

- 질문: 개별 target의 음성 cycle에는 다른 고장만 있는 cycle이 섞여 있는데, 이를 나누면 경보 해석이 달라지는가?
- 방법: 저장된 `12`, `14`, `15` cycle 결과를 재학습 없이 재집계해 완전 정상 오경보와 교차 고장 경보를 분리했다.
- 관련 파일: `research/16_fault_context_alert_analysis.py`, `research/outputs/16_fault_context_alert_analysis.md`, `research/outputs/16_fault_context_alert_summary.csv`

### 2026-08-27 - 17 보고서 근거 대조

- 방법: 8월 27일 초안 기준으로 저장 결과와 본문 수치·문장을 대조했고 122개 항목이 통과했다.
- 이후: 2026-11-08에 v0.7 기준으로 다시 작성했다. 아래 Timestamp 기록과 `21`, `22` 뒤에 실행한 결과가 현재 기록이다.

### 2026-11-08 - Timestamp 따옴표 수정과 재실행

- 문제: 863행의 `Timestamp`가 큰따옴표로 한 번 더 감싸여 있어 `to_datetime(errors="coerce")`에서 결측이 되어 구간 구성에서 빠졌다.
- 수정: `common.load_raw_data`에서 따옴표를 제거하고 00, 02, 03, 04, 09, 10, 12, 13, 14, 15, 16을 재실행했다. 01과 06은 시각 변환을 쓰지 않아 영향이 없고, 05, 07, 08, 11은 02·03·10의 구간을 쓰지만 재실행하지 않아 수정 전 값이다. 이 값들은 v0.7 본문에 쓰지 않았다.
- 결과: `cycle_run` 202개에서 231개로, 10-step window 4,035개에서 4,667개로, 완전 정상 cycle 115개에서 127개로 바뀌었다. 전후 비교는 `research/2026-11-08_timestamp_fix_comparison.md`에 있다.
- 작업 위치: 복제본 `UR3_Cobot_ML_tsfix`의 `research/timestamp-quote-fix` 브랜치. 이 기록 시점에 `main`에는 병합하지 않았다.

### 2026-11-08 - 18~20 보고서 그림과 Word 생성

- 방법: 그림 2(`18`)와 그림 1(`19`)을 저장 결과와 고정 구조로 그리고, `20`이 Markdown 원고를 2단 Word 파일로 만든다. 새 모델을 학습하지 않았다.
- 관련 파일: `research/18_detection_false_alarm_figure.py`, `research/19_study_design_figure.py`, `research/20_build_report_docx.py`

### 2026-11-08 - 21 입력 통제 비교와 Random Forest 시드 (사후)

- 질문: 행과 구간의 비교(표 1)는 평가 단위와 라벨이 달라 시계열 맥락의 효과로 읽을 수 없다. 같은 구간과 라벨로 평가하며 입력을 바꾸면 어떤가? Random Forest 결과는 시드에 얼마나 흔들리는가?
- 사전 고정: 없다. v0.6 검토 뒤 결과를 알고 추가한 사후 분석이며, 사전등록 문서의 Random Forest 시드 42 주 결과를 바꾸지 않는다.
- 공통 조건: 4,667개 10-step window와 window 라벨, 9개 block, SMOTE, 300 trees, `score > 0.50`.
- 입력: `rf_last_step_19`는 window의 마지막 행 19개 값, `rf_row_max_19`는 행 라벨로 학습한 뒤 window의 10개 행 점수 중 최댓값, `rf_19_raw_window`는 기존 133개 통계. 마지막 행은 처음 계획한 조건이었으나 양성 window 중 마지막 행도 양성인 비율이 23.0%, 11.3%, 36.6%라 입력이 고장 행을 보지 못하는 경우가 많아, `rf_row_max_19`를 같은 실행에 추가했다.
- 실행 범위: 입력 조건별 27회, 시드 43, 44 각 27회. 시드 42 통계 조건 27회는 재현 확인용으로 다시 돌렸고 `12` 저장 점수와 14,001개 window 모두 일치했다.
- 탐지/완전 정상 오경보(마지막 행, 행 최댓값, 통계): `System_Failure` 92/21, 98/11, 104/5. `ProtectiveStop` 70/15, 69/1, 74/2. `GripLost` 32/24, 39/6, 45/2.
- 시드: 2/3 합의에서 탐지는 104/104, 74/76, 45/48로 같고 완전 정상 오경보는 4/127, 2/127, 1/127, 교차 경보는 1/28, 3/56이다.
- 한계: 시드 42 한 번의 사후 비교이고 차이가 몇 건 수준이다. 133개 통계 대부분은 시점 순서를 쓰지 않으므로 순서 정보의 효과로 단정하지 않는다.
- 관련 파일: `research/21_rf_control_checks.py`, `research/outputs/21_rf_control_checks.md`, `research/outputs/21_rf_control_summary.csv`, `research/outputs/21_rf_control_window_predictions.csv`

### 2026-11-08 - 22 저장 결과 사후 점검

- 방법: 재학습 없이 저장 예측과 원자료에서 블록 구성, 결측·이어 붙인 구간 민감도, 같은 오경보 수준의 CNN 탐지와 사이클 AUC, McNemar, Autoencoder 점수와 움직임의 기술통계, 고장 전 예측의 전부 양성 F1 기준선을 재집계했다.
- 주의: 같은 오경보 비교는 시험 점수로 그린 곡선 위의 한 점이다. Autoencoder 기술통계는 인과를 확인하지 않는다.
- 관련 파일: `research/22_posthoc_saved_result_checks.py`, `research/outputs/22_posthoc_saved_results.md`

### 2026-11-08 - 23 구간 길이 민감도 (사후)

- 질문: 구간 길이를 5, 20-step으로 바꾸면 Random Forest와 1D CNN 비교의 결론이 달라지는가?
- 사전 고정: 없다. 결과를 알고 추가한 사후 분석이며 10-step을 주 결과로 유지한다. Autoencoder는 탐지가 낮아 제외했다.
- 공통 조건: `12`와 같은 19개 센서, 9개 block, 구간 라벨, `score > 0.50`, Random Forest 시드 42, 1D CNN 시드 42, 43, 44와 2/3 합의.
- 실행 범위: Random Forest 54회, 1D CNN 162회, 그리고 경로 확인용 10-step CNN 1회. 10-step 구간 재구성은 저장 캐시와 일치했고, 확인용 fold의 점수 차이는 0이었다. 실행 환경은 `12`와 같다.
- 데이터: 구간은 5-step 5,822개, 20-step 2,365개다. 20-step에서는 행이 부족한 완전 정상 cycle 1개가 빠져 230개 cycle이다.
- 탐지/완전 정상 오경보(5, 10, 20-step): Random Forest `System_Failure` 103/5, 104/5, 102/9. `ProtectiveStop` 72/7, 74/2, 70/5. `GripLost` 46/2, 45/2, 43/1. 1D CNN `System_Failure` 103/56, 104/23, 82/7. `ProtectiveStop` 74/25, 73/17, 65/12. `GripLost` 48/88, 48/27, 46/6.
- 해석: Random Forest는 길이에 비교적 안정적이었고, 1D CNN의 오경보와 탐지는 길이에 크게 의존했다. 10-step에서 본 오경보 차이를 1D CNN의 고유한 한계로 일반화하지 않는다. 5에서 10-step으로 늘리면 여섯 경우 모두 탐지 변화가 2건 이내이고 오경보가 같거나 줄었으며, 10에서 20-step으로 늘리면 여섯 경우 모두 탐지가 2~22건 줄었다. 세 길이 중에서는 10-step이 절충점이다.
- 한계: 같은 시험 block에서 세 길이만 본 사후 비교이므로 최적 길이로 단정하지 않는다. 1D CNN 구조를 길이마다 다시 맞추지 않았다. 길이마다 양성 비율이 달라 구간 지표는 비교하지 않는다. cycle당 구간이 평균 25.2, 20.2, 10.3개로 줄어 경보 기회도 줄어든다. cycle 행 수 중앙값은 24개이고 30행 이상인 cycle은 20.3%다.
- 관련 파일: `research/23_window_length_sensitivity.py`, `research/outputs/23_window_length_sensitivity.md`, `research/outputs/23_window_length_summary.csv`, `research/outputs/23_cycle_rows.csv`

### 2026-11-08 - v0.8 정합성 수정과 길이별 같은 오경보 비교

- 정정: 블록별 오경보 최댓값 중 `ProtectiveStop`과 `GripLost`는 완전 정상이 아니라 해당 고장이 없는 cycle을 분모로 계산되어 있었다. `22`에서 완전 정상 기준으로 다시 계산해 Random Forest 11.1%, 11.1%, 9.1%, 1D CNN 71.4%, 57.1%, 72.7%로 바꾸었다. `12_matched_consensus_summary.csv`의 기존 열은 정의가 다르므로 보고서에 쓰지 않는다.
- 표현: 행 최댓값 모델은 행 라벨로 학습해 학습 목표도 다르므로, 표 6을 입력만 바꾼 비교로 쓰지 않는다.
- 확인: `04`의 설정(5/10, 10/3, 5/3)은 2026-07-09 `03` 단일 split에서 고른 뒤 코드에 고정한 값이다. 현재 저장된 `03` best는 10/3, 10/5, 5/3이지만 설정을 다시 고르지 않는다.
- 사후 추가: `23`의 예측으로 1D CNN 임계값을 Random Forest의 0.50 기준 오경보 수에 맞춘 탐지를 계산했다. 재학습은 없다. 5, 10, 20-step에서 1D CNN과 Random Forest는 `System_Failure` 96, 101, 85와 103, 104, 102, `ProtectiveStop` 71, 68, 64와 72, 74, 70, `GripLost` 29, 37, 9와 46, 45, 43이다.
- 한계: 시험 점수로 맞춘 값이다. 20-step `GripLost`는 오경보 수 1, 2, 3건에서 Random Forest 43, 44, 45, 1D CNN 9, 44, 46으로 순서가 바뀌고, 5-step `ProtectiveStop`은 오경보 0~4건에서 1D CNN 탐지가 더 많다. 전체 곡선의 우열을 뜻하지 않는다.
- 관련 파일: `research/outputs/22_block_false_alarm.csv`, `research/outputs/23_same_false_alarm.csv`, `research/outputs/23_false_alarm_curve.csv`

### 2026-11-09 - 24, 25 일반화 범위 점검과 v0.9 (사후)

- 질문: 표 2의 탐지율과 모델 순서는 어디까지 말할 수 있는가? 오토인코더의 낮은 탐지는 방식의 한계인가, 학습 조건의 결과인가?
- 사전 고정: 없다. 결과를 알고 추가한 사후 분석이며 주 결과를 대체하지 않는다. 표의 숫자는 바꾸지 않았다.
- 기록 시간: 7,409행이 모두 2022-10-26이고 블록이 바뀌는 지점의 간격은 최대 2.0초다. 긴 공백은 사이클 105 안의 5.2시간 한 번이다. 블록은 수집 세션이 아니라 이어진 기록을 번호로 자른 근사 단위다.
- 시간순 분할(Random Forest 6회): 공백 앞 103개 cycle로 학습, 뒤 cycle로 평가. 탐지 53/71, 7/63, 0/25, 완전 정상 오경보 0/56, cycle AUC 0.9935, 0.9984, 0.9832. 온도를 빼면 탐지 70/71, 55/63, 22/25, 오경보 2/56, 0/56, 5/56. 분할이 하나뿐이고 1D CNN은 이 점검에 넣지 않았다.
- 사이클 길이: 27행 이상을 경보로 하면 `ProtectiveStop` 76/76, 오경보 1/127, 교차 6/28, AUC 0.9938이고 시험 블록 밖에서 기준을 골라도 같다. `System_Failure` 82/104, `GripLost` 26/48. Random Forest 첫 경보 구간의 끝은 첫 보호 정지 행보다 중앙값 5행 뒤다.
- 같은 자료의 5겹 Macro F1(Random Forest 60회), 랜덤→사이클: 행 0.7870→0.7239, 0.8169→0.7648, 0.7562→0.6843. 구간 0.9459→0.8500, 0.9609→0.8918, 0.9405→0.8157. `01`의 비교는 시험 행이 달라 방향을 말할 수 없다.
- 오토인코더 에폭: 최대 40의 27회 중 16회가 상한에 닿았고(1D CNN은 81회 중 1회), 9개 분할 모두에서 검증 손실이 센서를 구간 평균으로 바꾼 값보다 컸다. 400 고정(27회, GPU)에서 window ROC-AUC 0.8053, 0.8969, 0.6070, 탐지 46/104, 40/76, 8/48, 오경보 11/127, 교차 5/28, 30/56. `25`의 같은 집계가 저장된 40 상한 결과를 그대로 재현했다.
- 해석: 표 2의 100%는 블록 평가 조건의 값이고, 오토인코더의 낮은 탐지를 방식의 한계로 일반화하지 않는다. 400에서도 Random Forest보다 탐지가 적고 오경보가 많다.
- 문장 수정: 참고문헌 원문과 대조해 [1]의 25회 반복(시험 시나리오 기준), SMOTE-ENN과 K겹 언급, [5]의 RUL 정의 서술을 고쳤다. `수집 블록` 표현을 `25사이클 블록`으로 바꾸고, Timestamp 수정과 재실행, 계획서 질문 수, 구간 재현율과 정밀도, 파지 상실의 전류 해석, 보호 정지의 고장 전 사이클 6/85를 본문에 적었다.
- 관련 파일: `research/24_autoencoder_fixed_epochs.py`, `research/25_posthoc_generalization_checks.py`, `research/outputs/24_ae_fixed_epoch_runs.csv`, `research/outputs/24_ae_fixed_epoch_scores.csv`, `research/outputs/25_*.csv`, `research/outputs/25_posthoc_generalization_checks.md`

### 2026-11-09 - 주 비교 재실행과 영문 정리

- 질문: 공개 태그만으로 표 2, 3의 주 비교가 재현되는가?
- 실행: `submission-2026-11-12`(91b277e)를 GitHub에서 새로 받은 별도 폴더에서 `10_prepare_sequence_data`, `12_matched_rf_baseline`(27회), `12_matched_torch_models`(1D CNN 81회, 오토인코더 27회), `12_matched_results`, `13`, `16`을 차례로 실행했다.
- 결과: 구간 캐시 3개가 저장본과 같고, Random Forest 14,001행과 딥러닝 168,012행의 점수 차이 최댓값이 0.0, 판정이 다른 행은 0, 고른 에폭도 같다. 표 2, 3의 원천 값도 같다. Git에서 바뀐 파일은 실행 시간이 적힌 run 파일 2개와 비교 문서 1개뿐이다.
- 한계: 같은 PC와 같은 환경의 재실행이며 다른 하드웨어의 재현은 아니다. 센서 제거, 고전 모델, 사후 분석은 다시 학습하지 않았다.
- 문서: 영문 초록과 표 머리글, 표 주석, 캡션의 표현을 고쳤다. 외부 사전등록으로 읽힐 수 있는 표현을 `prespecified`로 바꾸고, 표의 숫자와 국문 본문은 그대로 두었다.

## 공통 기준

- 재현에 필요한 경로, 스크립트 버전, 주요 파라미터를 남긴다.
- 실패한 실험도 삭제하지 않고 실패 원인과 다음 판단을 기록한다.
- 사람의 검토가 필요한 해석은 중앙 검토 대기열에 연결한다.
