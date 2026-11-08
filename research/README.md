# UR3 CobotOps 연구 작업 공간

> [!note] 문서 역할
> 이 문서는 ML 로봇팔 연구의 프로젝트 기준 문서로 사용한다. 상태가 달라졌을 때만 갱신하고, 실험 세부 결과는 `EXPERIMENT_LOG.md`와 `outputs/` 아래 결과 파일에 남긴다.

이 폴더는 기말 프로젝트 원본 노트북을 직접 수정하지 않고, 학술연구장학 연구 실행을 위해 만든 재현용 작업 공간이다.

## 목적

- 기존 UR3 CobotOps 기말 프로젝트의 분류 기반 이상탐지 baseline을 재현한다.
- row 단위 분류의 한계를 확인하고, cycle-aware split과 cycle 내부 window feature로 시계열 맥락을 반영한다.
- 구간 단위 이상탐지와 고장 발생 전 예측 실험을 분리하여 해석한다.
- 공개 센서만으로 논문의 공정조건 수준을 cycle별로 구분할 수 있는지 증거 범위를 진단한다.

## 확정 범위

- 포함: 공개 `ur3_cobotops.csv`, `System_Failure`, `ProtectiveStop`, `GripLost`, `Random Forest`, `SMOTE + Random Forest`, cycle-aware split, window feature baseline.
- 포함: 고정 10-step raw sequence를 입력으로 사용하는 소형 1D CNN과 단층 LSTM 비교.
- 제외: 공개 CSV에 없는 `workload`, `movement speed`, `gripping force`를 직접 feature처럼 해석하는 분석, 실제 로봇팔 제어 시스템 구현, 원본 데이터의 Obsidian Vault 복사.

## 현재 파일

- `common.py`: 데이터 로딩, 컬럼 정리, 타깃 생성, 공통 feature 정의
- `00_data_audit.py`: 데이터셋 구조, 결측치, 타깃 분포, Timestamp/cycle 구조 점검
- `01_baseline_reproduction.py`: row 단위 baseline과 split 방식 비교
- `02_window_feature_baseline.py`: cycle_run 내부 window feature 기반 분류 baseline
- `03_pre_failure_window_baseline.py`: first positive 이후 데이터를 제외한 고장 발생 전 window 분류 baseline
- `04_pre_failure_repeated_split_validation.py`: pre-failure 대표 조합의 반복 cycle-run-group split 검증
- `05_pre_failure_threshold_sensitivity.py`: pre-failure 대표 조합의 threshold 민감도 검증
- `06_process_condition_separability.py`: 공정조건 3개 수준의 latent cluster 및 25-cycle 후보 블록 분리 가능성 진단
- `07_cycle_run_revalidation.py`: raw cycle 경계 결과와 cycle_run 수정 결과 비교
- `08_block_held_out_robustness.py`: 25-cycle acquisition block held-out 구간 이상탐지 강건성 검증
- `09_event_level_block_validation.py`: 공통 10-step·동일 block 분할의 사건 단위 탐지율과 정상 cycle 오경보 검증
- `10_prepare_sequence_data.py`: 기존 `09`와 동일한 10-step window를 10×26 raw sequence로 준비하고 정합성 검증
- `10_torch_sequence_models.py`: 외부 PyTorch 환경에서 고정 1D CNN/LSTM을 block-held-out 방식으로 학습
- `10_sequence_model_results.py`: Random Forest와 sequence model의 window·event·오경보 지표 비교
- `11_sequence_error_analysis.py`: 기존 예측에서 seed 반복 오류, block 집중도, 모델 간 오류 겹침, 정상 cycle 센서 차이 분석
- `2026-08-23_project_to_research_synthesis.md`: 수업 프로젝트와 학술연구의 차이, 현재 결론, 한계, 계획서 이행 상태를 정리한 종합 문서
- `2026-08-23_lstm_autoencoder_preregistration.md`: 동일 19개 센서 비교와 정상-only LSTM Autoencoder의 분석 기준을 결과 확인 전에 고정한 문서
- `12_matched_rf_baseline.py`: 10-step×19개 원본 센서를 133개 통계 feature로 요약한 고정 Random Forest 기준선
- `12_matched_torch_models.py`: 같은 10-step×19개 원본 센서를 사용하는 1D CNN과 정상-only LSTM Autoencoder 학습
- `12_matched_results.py`: 3-seed cycle consensus, Wilson 95% 신뢰구간, threshold 민감도와 모델 간 오류 비교 집계
- `13_final_evaluation_tables.py`: 저장된 `12` prediction에서 최종 confusion matrix와 ROC-AUC·PR-AUC 표를 재집계
- `2026-08-23_sensor_group_ablation_preregistration.md`: 센서 그룹 ablation의 비교 조건과 해석 기준을 결과 확인 전에 고정한 문서
- `14_sensor_group_ablation.py`: 동일 Random Forest 조건에서 센서 그룹 10개 변형을 비교하는 실행 코드
- `14_sensor_group_ablation_results.py`: Window·cycle·block 지표와 paired error를 집계하는 코드
- `2026-08-23_classical_model_comparison_preregistration.md`: 동일 시계열 특징 기반 기본 분류 모델 비교 조건을 결과 확인 전에 고정한 문서
- `15_classical_model_comparison.py`: Logistic Regression과 RBF SVM의 9-block held-out 실행 코드
- `15_classical_model_comparison_results.py`: 기존 Random Forest를 포함한 window·cycle·block 지표와 paired error 집계 코드
- `16_fault_context_alert_analysis.py`: 기존 cycle 결과에서 완전 정상 오경보와 다른 고장에 대한 교차 경보를 분리하는 파생 분석 코드
- `2026-08-26_related_work_review.md`: UR3 CobotOps 직접 활용 연구와 방법론 참고 논문의 근거 수준을 구분한 문헌 검토
- `2026-08-27_research_report_draft.md`: 현재 결과와 한계를 논문 형식으로 정리한 연구보고서 초안
- `2026-08-27_reproducibility_review_plan.md`: 내부 수치 대조, CPU·GPU 전체 재실행, 문체·출력 검토를 분리한 재현성 점검 계획
- `21_rf_control_checks.py`: 같은 구간과 라벨에서 입력을 바꾼 Random Forest와 추가 시드를 실행하는 사후 점검 코드
- `22_posthoc_saved_result_checks.py`: 저장 결과에서 블록 구성, 결측 민감도, 같은 오경보 비교, Autoencoder 기술통계, 고장 전 기준선을 재집계하는 코드
- `23_window_length_sensitivity.py`: Random Forest와 1D CNN을 5, 20스텝으로 다시 학습하는 사후 민감도 코드
- `17_report_evidence_validation.py`: 보고서 핵심 수치와 저장된 결과물의 일치 여부를 검사하는 코드
- `run_report_validation.ps1`: 결과 재집계와 보고서 수치 대조를 한 번에 실행하는 스크립트
- `requirements.txt`: 연구 스크립트를 실제 실행한 Python 패키지 버전
- `EXPERIMENT_LOG.md`: 실제 실행한 실험의 시간순 기록
- `outputs/`: 실행 결과 CSV/Markdown 저장 위치

## 11월 12일 제출 현황

2026년 11월 8일 기준 최신 작업본은 [v0.7](2026-11-08_research_report_v0.7.md)이며 `research/timestamp-quote-fix` 브랜치에 있다. `main`과 `research/sensor-ablation`에는 Timestamp 수정 전 결과(`cycle_run` 202개)가 남아 있고, 수정 전후 차이는 [비교 문서](2026-11-08_timestamp_fix_comparison.md)에 있다. 이 저장소에서 제출 완료 기록은 확인되지 않았다. 마감은 11월 12일 17:00이고 공지에 정해진 양식이나 분량은 없다.

### 확인된 결과

- 공개 UR3 CobotOps의 `cycle_run` 231개에서 경계를 넘지 않는 10-step window 4,667개를 만들고, 같은 19개 센서로 9개 수집 블록을 하나씩 제외해 평가했다. 주 결과는 이상이 포함된 구간의 탐지 성능이다.
- Random Forest의 고장 cycle 재현율과 완전 정상 cycle 오경보율은 `System_Failure` 100.00%와 3.94%, `ProtectiveStop` 97.37%와 1.57%, `GripLost` 93.75%와 1.57%였다. 1D CNN은 재현율이 비슷했으나 완전 정상 오경보율이 13.39~21.26%였고, 정상 데이터로만 학습한 LSTM Autoencoder의 q95 재현율은 0~6.58%였다. [주 결과 및 경보 분해](outputs/16_fault_context_alert_analysis.md)
- 기본 분류 모델 비교와 센서 그룹 제거 실험까지 수행했다. `GripLost`의 고장 전 분석은 평균 재현율 28.14%의 탐색 결과로, 실용적인 조기경고 성능은 확인하지 못했다. [기본 모델 비교](outputs/15_classical_model_comparison.md), [센서 그룹 비교](outputs/14_sensor_group_ablation.md), [고장 전 분석](outputs/04_pre_failure_repeated_split_results.md)
- 같은 구간과 라벨에서 입력만 바꾼 사후 비교(`21`)와 저장 결과의 사후 점검(`22`), 구간 길이 민감도(`23`)를 추가했다. 사전 문서에 없던 분석이며 주 결과를 대체하지 않는다. [사후 RF 비교](outputs/21_rf_control_checks.md), [사후 점검](outputs/22_posthoc_saved_results.md), [구간 길이 민감도](outputs/23_window_length_sensitivity.md)
- `17`은 v0.7 본문과 표의 수치를 저장 결과에서 다시 만들어 대조한다. 검사 항목이 모두 통과했다. [내부 검사 기록](outputs/17_report_evidence_validation.md)

### 빠진 근거와 초안 수정점

- 원본 CSV부터 별도 작업공간에서 CPU 모델과 GPU 모델을 다시 학습해 예측 및 지표를 대조한 완료 기록은 확인되지 않았다. 최종 초안에는 제출 시점에 실제로 완료한 검증 범위를 적는다. [재현 절차 및 판정 기준](2026-08-27_reproducibility_review_plan.md)
- v0.7의 커밋 해시는 제출 커밋이 정해진 뒤 적는다.
- 공정조건별 일반화에는 cycle별 운전조건 대응표가 필요하고, 다른 로봇으로의 일반화에는 별도 데이터가 필요하다. 두 근거 모두 현재 확보된 자료에서는 확인되지 않았다.

## 실행 순서

```powershell
python -X utf8 research\00_data_audit.py
python -X utf8 research\01_baseline_reproduction.py
python -X utf8 research\02_window_feature_baseline.py
python -X utf8 research\03_pre_failure_window_baseline.py
python -X utf8 research\04_pre_failure_repeated_split_validation.py
python -X utf8 research\05_pre_failure_threshold_sensitivity.py
python -X utf8 research\06_process_condition_separability.py
python -X utf8 research\07_cycle_run_revalidation.py
python -X utf8 research\08_block_held_out_robustness.py
python -X utf8 research\09_event_level_block_validation.py
python -X utf8 research\10_prepare_sequence_data.py
& "D:\Projects\Cube_Codex\.venvs\sam2_clean\Scripts\python.exe" -X utf8 research\10_torch_sequence_models.py --manifest research\.sequence_cache\10_sequence_manifest.json --output-dir research\outputs --device cuda
python -X utf8 research\10_sequence_model_results.py
python -X utf8 research\11_sequence_error_analysis.py
python -X utf8 research\12_matched_rf_baseline.py
& "D:\Projects\Cube_Codex\.venvs\sam2_clean\Scripts\python.exe" -X utf8 research\12_matched_torch_models.py --manifest research\.sequence_cache\10_sequence_manifest.json --output-dir research\outputs --device cuda
python -X utf8 research\12_matched_results.py
python -X utf8 research\13_final_evaluation_tables.py
python -X utf8 research\14_sensor_group_ablation.py
python -X utf8 research\14_sensor_group_ablation_results.py
python -X utf8 research\15_classical_model_comparison.py
python -X utf8 research\15_classical_model_comparison_results.py
python -X utf8 research\16_fault_context_alert_analysis.py
python -X utf8 research\21_rf_control_checks.py
python -X utf8 research\22_posthoc_saved_result_checks.py
python -X utf8 research\23_window_length_sensitivity.py rf
& "D:\Projects\Cube_Codex\.venvs\sam2_clean\Scripts\python.exe" -X utf8 research\23_window_length_sensitivity.py cnn
python -X utf8 research\23_window_length_sensitivity.py summary
python -X utf8 research\17_report_evidence_validation.py
```

보고서 수치만 다시 집계하고 대조할 때는 다음 스크립트를 사용한다.

```powershell
powershell -ExecutionPolicy Bypass -File research\run_report_validation.ps1
```

## 데이터 provenance 및 실행 환경

- 공식 출처: UCI Machine Learning Repository, UR3 CobotOps, DOI `10.24432/C5J891`, 데이터 라이선스 `CC BY 4.0`.
- `dataset/ur3_cobotops.csv` SHA-256: `C789CDA10ACB354A7C1689F617D94A5F39A93FD8CB6C004AD16D36CEA55A74A3`.
- `dataset/dataset_02052023.xlsx` SHA-256: `F0F10917DE9056908A82CEC1AA459DDF9DA2D2DB70D269B128F99241C8796091`.
- CSV와 Excel은 7,409행 × 24열 구조가 같고, 수치 차이는 최대 약 `4.4e-8`로 반올림 수준이다.
- IF-FCM 참고 논문 DOI: `10.1007/978-3-031-63851-0_6`.
- `09` 검증 환경: Python 3.12.3, NumPy 2.4.3, pandas 3.0.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1, openpyxl 3.1.5.
- `10` sequence model 환경: `D:\Projects\Cube_Codex\.venvs\sam2_clean`, Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Ti. 기존 환경의 패키지는 변경하지 않았다.
- `research/requirements.txt`는 데이터 준비와 결과 집계 환경을 기록하며 PyTorch는 위 외부 환경에서 재사용한다.
- 논문의 RTDE 125 Hz 설명은 인터페이스 동작 주파수이며, 공개 CSV의 양수 Timestamp 간격 중앙값은 약 1.005초다. Window 크기는 초가 아니라 step으로 해석한다.

## 현재 연구상 주의점

이 절의 수치 중 다시 확인하지 않은 값은 Timestamp 수정 전일 수 있다. 최신 수치는 v0.7과 `outputs/`를 기준으로 한다.

- 논문에서 말하는 `workload`, `movement speed`, `gripping force`는 공개 CSV의 직접 컬럼이 아니다.
- 원본 `cycle` ID 224, 225, 226, 227, 229, 230, 231, 232, 233은 시간상 떨어진 두 시행 구간에 재등장한다. 현재 `02`-`05`는 `cycle_run` 기준으로 수정·재실행됐다.
- 공정조건별 3-cluster를 실제 1/2/3 kg, 60/80/100%, 80/100/120 N 정답 라벨로 간주하면 안 된다.
- 25-cycle 후보 블록은 온도 feature만으로 매우 잘 구분되어, 공정조건과 thermal/session drift가 혼재할 가능성이 크다.
- 따라서 본 연구는 해당 공정 조건을 직접 검증하기보다, 센서 시계열에 남아 있는 이상탐지 패턴을 분석한다.
- `System_Failure`는 `ProtectiveStop OR GripLost`로 정의하되, 두 고장 유형은 별도 타깃으로도 비교한다.
- row 단위 random split은 성능을 낙관적으로 보일 수 있으므로, `cycle_run_group` 및 acquisition block held-out 결과와 비교한다.
- 시계열 맥락은 `cycle_run` 경계를 넘지 않는 5, 10, 20 step window feature로 반영한다.
- `02`는 구간 단위 이상탐지이고, `03`은 first positive 이전 데이터만 쓰는 고장 발생 전 예측 실험이므로 성능을 직접 비교하지 않는다.
- `04` 반복 split 기준으로 `GripLost`는 positive를 잡는 결과가 상대적으로 안정적이고, `ProtectiveStop`과 `System_Failure`는 해석에 더 주의가 필요하다.
- `05` threshold 조정은 recall 개선 가능성을 보지만, false positive 부담이 함께 증가하므로 최종 성능 주장으로 바로 쓰지 않는다.
- `08`은 구간 단위 이상탐지에서 9개 acquisition block 모두 positive를 잡았지만, 이미 이상이 포함된 window 탐지이므로 pre-failure 결과로 해석하지 않는다.
- `09`의 event cycle 탐지는 실제 positive가 포함된 window에서의 탐지 여부이며, 고장 전 경고 성능이 아니다.
- `09`~`15`에서 `normal_cycle_false_alarm_rate`로 저장된 값은 target이 없는 모든 cycle을 분모로 사용한다. 개별 target에서는 다른 고장만 발생한 cycle도 포함되므로 정확한 명칭은 `target-negative 경보율`이다.
- `16`은 기존 prediction을 바꾸지 않고 완전 정상 cycle 오경보율과 다른 고장만 있는 cycle의 교차 고장 경보율을 분리했다. 이후 최종 서술은 이 정정 지표를 우선한다.
- temperature 제거는 `System_Failure`에서 개선됐지만 target별 효과가 일관되지는 않아 ablation 결과를 함께 보고한다.
- `10`의 Random Forest는 10-step을 182개 통계 feature로 요약하고, 1D CNN/LSTM은 같은 구간을 10×26 순서 입력으로 사용한다.
- `10`의 deep learning 결과는 seed 3개 평균이며, Random Forest는 `09`의 고정 1회 prediction을 재사용한다.
- `10`의 window 지표는 seed별 9개 held-out prediction을 합친 pooled 지표라서, `09` 보고서의 block별 평균과 숫자가 직접 같지는 않다.
- `11`의 deep learning cycle consensus는 3개 seed 중 2개 이상이 같은 cycle에서 경보한 경우다. Sensor shift는 동일 window에 대한 2/3 consensus만 사용한다.
- `11`에서 deep learning 오경보의 대부분은 Random Forest와 겹치지 않았다. `System_Failure`의 deep-only 오경보는 1D CNN/LSTM 15/19개, `ProtectiveStop` 23/26개, `GripLost` 47/54개였다.
- 정상 cycle의 반복 오경보 window는 target별로 낮은 관절 움직임 범위, 특정 전류 수준, Tool current 변화 또는 온도 수준 차이를 보였으나, 겹치는 window의 기술통계이므로 인과적 feature importance로 해석하지 않는다.
- `12`는 모델 입력을 10-step×19개 원본 센서로 맞췄다. Random Forest만 각 센서의 mean, std, min, max, range, delta, slope를 계산해 133개 feature로 변환하며, 1D CNN과 LSTM Autoencoder는 같은 원본 순서를 직접 입력받는다.
- `12`의 Autoencoder는 `System_Failure=0`인 완전 정상 cycle만 학습했다. 정상 재구성 오차가 모든 고장 유형을 포괄하는 이상 점수라는 가정은 결과적으로 지지되지 않았다.
- Autoencoder calibration 정상 cycle은 outer fold별 13~38개로 적다. q90, q95, q97.5 민감도에서도 결론이 바뀌지 않았으며, 더 유리한 threshold를 주 결과로 교체하지 않는다.
- `12`의 cycle-level 표본 수는 전체 231개다. Wilson 신뢰구간은 비율의 불확실성만 나타내며 block/session 내부 상관이나 동일 데이터 재사용 문제를 해소하지 않는다.
- `13`은 저장된 `12` prediction을 최종 표 형식으로 재집계한 결과이며 독립적인 새 실험이 아니다.
- `14`의 `all_19`는 `12` Random Forest의 score와 prediction에 정확히 일치한다.
- `14`의 센서 제거·단독 사용은 feature 차원과 SMOTE 공간도 함께 바꾸므로 센서의 인과적 효과를 증명하지 않는다.
- `Tool_current`는 세 타깃에서 일관되게 유용하지 않았고, 온도 단독 탐지력은 낮지만 온도 제거 후 일부 정상 cycle 오경보가 증가했다.
- `15`는 Logistic Regression과 RBF SVM에만 학습 fold 내부 표준화를 적용했다. 거리·계수 기반 모델에 필요한 전처리 차이이며 입력 window와 133개 특징, SMOTE, test block은 같다.
- Logistic Regression의 높은 event recall은 완전 정상 cycle 26.77~47.24% 오경보와 함께 나타났으므로 우수한 탐지기로 단독 해석하지 않는다.
- RBF SVM은 `ProtectiveStop`에서 Random Forest보다 오경보 cycle이 2개 적었지만 event cycle을 1개 더 놓쳤다. 타깃별 trade-off이며 전체 우위가 아니다.
- `15`는 고정 `C=1`, RBF `gamma=scale`, threshold 0.50의 최소 비교다. 현재 결과를 본 뒤 모델별 tuning을 추가하지 않는다.

## 근거

- 코드 및 데이터: `dataset/ur3_cobotops.csv`, `research/common.py`, `research/00_data_audit.py`, `research/01_baseline_reproduction.py`, `research/02_window_feature_baseline.py`, `research/03_pre_failure_window_baseline.py`
- 결과물: `research/outputs/00_data_audit_report.md`, `research/outputs/01_baseline_results.md`, `research/outputs/02_window_feature_results.md`, `research/outputs/03_pre_failure_window_results.md`, `research/outputs/04_pre_failure_repeated_split_results.md`, `research/outputs/05_pre_failure_threshold_sensitivity_results.md`
- 공정조건 진단: `research/outputs/06_process_condition_separability.md`, `research/outputs/06_condition_cluster_summary.csv`, `research/outputs/06_condition_block_classification.csv`
- 경계·강건성 검증: `research/outputs/07_cycle_run_revalidation.md`, `research/outputs/08_block_held_out_robustness.md`, `research/outputs/09_event_level_block_validation.md`
- Sequence model 비교: `research/outputs/10_sequence_model_comparison.md`, `research/outputs/10_sequence_model_summary.csv`, `research/outputs/10_sequence_block_results.csv`, `research/outputs/10_sequence_window_predictions.csv`
- 오류 분석: `research/outputs/11_sequence_error_analysis.md`, `research/outputs/11_error_cycle_details.csv`, `research/outputs/11_error_block_summary.csv`, `research/outputs/11_false_alarm_sensor_shifts.csv`
- 동일 센서 후속 비교: `research/2026-08-23_lstm_autoencoder_preregistration.md`, `research/outputs/12_matched_lstm_autoencoder_comparison.md`, `research/outputs/12_matched_consensus_summary.csv`, `research/outputs/12_matched_pairwise_cycle_errors.csv`
- 최종 평가표: `research/outputs/13_final_evaluation_tables.md`, `research/outputs/13_cycle_consensus_confusion_metrics.csv`
- 센서 그룹 ablation: `research/2026-08-23_sensor_group_ablation_preregistration.md`, `research/outputs/14_sensor_group_ablation.md`, `research/outputs/14_sensor_group_ablation_summary.csv`, `research/outputs/14_sensor_group_ablation_paired_errors.csv`
- 기본 분류 모델 비교: `research/2026-08-23_classical_model_comparison_preregistration.md`, `research/outputs/15_classical_model_comparison.md`, `research/outputs/15_classical_model_summary.csv`, `research/outputs/15_classical_model_paired_errors.csv`
- Fault-context 경보 정정: `research/16_fault_context_alert_analysis.py`, `research/outputs/16_fault_context_alert_analysis.md`, `research/outputs/16_fault_context_alert_summary.csv`, `research/outputs/16_fault_context_cycle_results.csv`
- 관련 연구 검토: `research/2026-08-26_related_work_review.md`
- 연구 종합: `research/2026-08-23_project_to_research_synthesis.md`
- 연구보고서 초안과 검증: `research/2026-08-27_research_report_draft.md`, `research/2026-08-27_reproducibility_review_plan.md`, `research/outputs/17_report_evidence_validation.md`
- 관련 메모: `research/2026-07-08_time_series_method_decision.md`

## 열린 질문

- `GripLost` pre-failure threshold 0.30을 후속 검증 후보로 유지할지 정해야 한다.
- 공정조건 분류는 대응표를 확보하지 않는 한 연구 범위에 포함하지 않는다.

## 갱신 기록

- 2026-11-08: Timestamp 따옴표 수정 후 재실행 결과와 사후 점검(`21`, `22`, `23`)을 반영하고, `17`을 v0.7 기준으로 바꾸었다.
- 2026-07-10: 프로젝트 기준 문서로 지정하고, `03_pre_failure_window_baseline.py`까지 반영한 현재 상태를 정리했다.
- 2026-07-22: 반복 split 검증(`04`)과 threshold 민감도 검증(`05`) 결과를 반영했다.
- 2026-08-10: 공정조건 분리 가능성 진단(`06`)과 raw `cycle` ID 재등장 문제를 반영했다.
- 2026-08-10: `cycle_run` 기준으로 `02`-`05`를 재실행하고 경계 재검증(`07`)과 acquisition block held-out 검증(`08`)을 반영했다.
- 2026-08-12: 공통 10-step 사건 단위 검증(`09`)을 반영하고 데이터 provenance와 실행 환경을 기록했다.
- 2026-08-12: 외부 PyTorch 환경을 변경 없이 재사용해 고정 1D CNN/LSTM 비교(`10`)를 완료하고 결과와 한계를 반영했다.
- 2026-08-23: 새 학습 없이 `09`·`10` 예측을 재사용해 seed 반복 오류, block 집중도, 모델 간 오류 겹침과 정상 cycle 센서 차이 분석(`11`)을 완료했다.
- 2026-08-23: 결과 확인 전에 동일 19개 센서 비교를 사전등록하고, Random Forest·1D CNN·정상-only LSTM Autoencoder의 9-block·3-seed 후속 비교(`12`)를 완료했다.
- 2026-08-23: 수업 프로젝트의 성과와 과장 가능성, 학술연구에서 추가한 검증, 현재 한계와 최종 보고서 구조를 연구 종합 문서로 정리했다.
- 2026-08-23: 저장된 prediction에서 최종 confusion matrix와 ROC-AUC·PR-AUC 표(`13`)를 재집계했다.
- 2026-08-23: 결과 확인 전에 센서 그룹 비교를 사전등록하고, 동일 Random Forest 조건의 10개 입력 변형·270회 block 평가(`14`)를 완료했다.
- 2026-08-23: 결과 확인 전에 기본 분류 모델 비교를 사전등록하고, 동일 133개 시계열 특징에서 Logistic Regression·RBF SVM의 54회 block 평가(`15`)를 완료했다.
- 2026-08-26: 기존 `12`·`14`·`15` cycle 결과를 재학습 없이 재집계해 완전 정상 오경보와 교차 고장 경보를 분리(`16`)하고, UR3 CobotOps 직접 활용 연구와 방법론 참고 논문을 검토했다.
- 2026-08-27: 논문 형식 연구보고서 초안을 작성하고, 저장 결과와 본문 핵심 수치·문장 122개 항목의 내부 일치 여부를 검증했다. 전체 CPU·GPU 재실행과 최종 윤문·출력 검토는 후속 단계로 분리했다.
- 2026-11-07: 11월 12일 제출을 기준으로 작업본, 확인된 결과, 보완할 근거를 정리했다.
- 2026-11-08: Autoencoder 보정과 전류 계열 제거의 서술을 코드 및 저장 결과에 맞춰 수정한 v0.2를 작성했다.
- 2026-11-08: v0.3에서 연구 동기, EDA, 평가와 결과의 순서를 정리했다. 저장 결과를 사용해 주 결과표에 건수와 분모를 넣고 window 성능표와 모델 비교 그림을 추가했다. 최종 윤문과 자체 검토는 후속 단계로 둔다.
