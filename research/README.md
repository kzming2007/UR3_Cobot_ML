# UR3 CobotOps 연구 작업 공간

이 폴더는 기말 프로젝트 원본 노트북을 직접 수정하지 않고, 학술연구장학 연구 실행을 위해 만든 재현용 작업 공간이다.

## 목적

- 기존 UR3 CobotOps 기말 프로젝트의 분류 기반 이상탐지 baseline을 재현한다.
- row 단위 분류의 한계를 확인하고, cycle-aware split과 cycle 내부 window feature로 시계열 맥락을 반영한다.
- 구간 단위 이상탐지와 고장 발생 전 예측 실험을 분리하여 해석한다.
- 공개 센서만으로 논문의 공정조건 수준을 cycle별로 구분할 수 있는지 증거 범위를 진단한다.

## 폴더 구성

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
- `2026-08-23_project_to_research_synthesis.md`: 8월 기준으로 수업 프로젝트와 학술연구의 차이, 결론과 한계를 정리한 종합 문서. 수치는 Timestamp 수정 전 값이다
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
- `2026-08-27_research_report_draft.md`: 8월에 쓴 논문 형식 초안. 수치는 Timestamp 수정 전 값이고 최종 원고는 v0.9이다
- `2026-08-27_reproducibility_review_plan.md`: 내부 수치 대조, CPU·GPU 전체 재실행, 문체·출력 검토를 분리한 재현성 점검 계획
- `21_rf_control_checks.py`: 같은 구간과 라벨에서 입력을 바꾼 Random Forest와 추가 시드를 실행하는 사후 점검 코드
- `22_posthoc_saved_result_checks.py`: 저장 결과에서 블록 구성, 결측 민감도, 같은 오경보 비교, Autoencoder 기술통계, 고장 전 기준선을 재집계하는 코드
- `23_window_length_sensitivity.py`: Random Forest와 1D CNN을 5, 20스텝으로 다시 학습하는 사후 민감도 코드
- `24_autoencoder_fixed_epochs.py`: LSTM Autoencoder를 400 epoch로 고정해 다시 학습하는 사후 점검 코드
- `25_posthoc_generalization_checks.py`: 기록 시간, 긴 공백 전후의 시간순 분할, 사이클 길이 기준선, 같은 자료의 랜덤·사이클 5겹 비교, Autoencoder 에폭 상한 점검 코드
- `17_report_evidence_validation.py`: 보고서 핵심 수치와 저장된 결과물의 일치 여부를 검사하는 코드
- `18_detection_false_alarm_figure.py`: 저장 결과에서 고장 사이클 재현율과 완전 정상 오경보율 그림(그림 2)을 그리는 코드
- `19_study_design_figure.py`: 연구 설계 그림(그림 1)을 그리는 코드
- `20_build_report_docx.py`: 원고 Markdown을 2단 Word 파일로 만드는 코드
- `run_report_validation.ps1`: 결과 재집계와 보고서 수치 대조를 한 번에 실행하는 스크립트
- `requirements.txt`: 연구 스크립트를 실제 실행한 Python 패키지 버전
- `EXPERIMENT_LOG.md`: 실제 실행한 실험의 시간순 기록
- `outputs/`: 실행 결과 CSV/Markdown 저장 위치

## 최종 보고서

최종본은 [PDF](2026_학술연구장학_결과보고서_남궁도현_김택명.pdf)이고 원고는 [v0.9](2026-11-09_research_report_v0.9.md)이다. `research/sensor-ablation`에는 Timestamp 수정 전 결과(`cycle_run` 202개)가 남아 있고, 수정 전후 차이는 [비교 문서](2026-11-08_timestamp_fix_comparison.md)에 있다.

### 확인된 결과

- 공개 UR3 CobotOps의 `cycle_run` 231개에서 경계를 넘지 않는 10-step window 4,667개를 만들고, 같은 19개 센서로 사이클 번호를 25개씩 묶은 9개 블록을 하나씩 제외해 평가했다. 주 결과는 이상이 포함된 구간의 탐지 성능이다.
- Random Forest의 고장 cycle 재현율과 완전 정상 cycle 오경보율은 `System_Failure` 100.00%와 3.94%, `ProtectiveStop` 97.37%와 1.57%, `GripLost` 93.75%와 1.57%였다. 1D CNN은 재현율이 비슷했으나 완전 정상 오경보율이 13.39~21.26%였고, 정상 데이터로만 학습한 LSTM Autoencoder의 q95 재현율은 사전 조건인 최대 40 epoch에서 0~6.58%였다. [주 결과 및 경보 분해](outputs/16_fault_context_alert_analysis.md)
- 기본 분류 모델 비교와 센서 그룹 제거 실험까지 수행했다. `GripLost`의 고장 전 분석은 평균 재현율 28.14%의 탐색 결과로, 실용적인 조기경고 성능은 확인하지 못했다. [기본 모델 비교](outputs/15_classical_model_comparison.md), [센서 그룹 비교](outputs/14_sensor_group_ablation.md), [고장 전 분석](outputs/04_pre_failure_repeated_split_results.md)
- 같은 구간과 라벨로 평가한 사후 입력 비교(`21`)와 저장 결과의 사후 점검(`22`), 구간 길이 민감도(`23`), 일반화 범위 점검(`24`, `25`)을 추가했다. 사전 문서에 없던 분석이며 주 결과를 대체하지 않는다. [사후 RF 비교](outputs/21_rf_control_checks.md), [사후 점검](outputs/22_posthoc_saved_results.md), [구간 길이 민감도](outputs/23_window_length_sensitivity.md), [일반화 범위 점검](outputs/25_posthoc_generalization_checks.md)
- `17`은 v0.9 본문과 표의 수치를 저장 결과에서 다시 만들어 대조한다. 검사 항목이 모두 통과했다. [내부 검사 기록](outputs/17_report_evidence_validation.md)

### 재현 범위

- 2026-11-09에 태그를 새로 받은 복제본에서 원본 CSV의 전처리부터 주 비교 135회를 다시 학습했고, 점수와 판정이 저장 결과와 같았다. 같은 PC와 환경에서 실행한 것이며, 센서 제거, 고전 모델, 사후 분석은 다시 학습하지 않았다. [재현 절차 및 판정 기준](2026-08-27_reproducibility_review_plan.md)
- v0.9 본문의 저장소 주소는 `submission-2026-11-12` 태그를 가리킨다.
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
& "D:\Projects\Cube_Codex\.venvs\sam2_clean\Scripts\python.exe" -X utf8 research\24_autoencoder_fixed_epochs.py
python -X utf8 research\25_posthoc_generalization_checks.py
python -X utf8 research\17_report_evidence_validation.py
```

보고서 수치만 다시 집계하고 대조할 때는 다음 스크립트를 사용한다.

```powershell
powershell -ExecutionPolicy Bypass -File research\run_report_validation.ps1
```

## 데이터 출처와 실행 환경

- 공식 출처: UCI Machine Learning Repository, UR3 CobotOps, DOI `10.24432/C5J891`, 데이터 라이선스 `CC BY 4.0`.
- `dataset/ur3_cobotops.csv` SHA-256: `C789CDA10ACB354A7C1689F617D94A5F39A93FD8CB6C004AD16D36CEA55A74A3`.
- `dataset/dataset_02052023.xlsx` SHA-256: `F0F10917DE9056908A82CEC1AA459DDF9DA2D2DB70D269B128F99241C8796091`.
- CSV와 Excel은 7,409행 × 24열 구조가 같고, 수치 차이는 최대 약 `4.4e-8`로 반올림 수준이다.
- IF-FCM 참고 논문 DOI: `10.1007/978-3-031-63851-0_6`.
- `09` 검증 환경: Python 3.12.3, NumPy 2.4.3, pandas 3.0.3, scikit-learn 1.8.0, imbalanced-learn 0.14.1, openpyxl 3.1.5.
- `10` sequence model 환경: `D:\Projects\Cube_Codex\.venvs\sam2_clean`, Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3060 Ti. 기존 환경의 패키지는 변경하지 않았다.
- `research/requirements.txt`는 데이터 준비와 결과 집계 환경을 기록하며 PyTorch는 위 외부 환경에서 재사용한다.
- 논문의 RTDE 125 Hz 설명은 인터페이스 동작 주파수이며, 공개 CSV의 양수 Timestamp 간격 중앙값은 약 1.005초다. Window 크기는 초가 아니라 step으로 해석한다.
