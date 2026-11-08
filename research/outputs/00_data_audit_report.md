# 00 데이터 감사

- Dataset: `C:\Users\PC-001\Desktop\ML 프로젝트\UR3_Cobot_ML_tsfix\dataset\ur3_cobotops.csv`
- 연구용 타깃/파생 feature 추가 후 크기: 7,409 rows x 34 columns
- 원본 기준 `dropna()` 후 행 수: 7,355
- cycle 범위: 1 - 264 (240 unique cycles)

## 타깃 분포

| target | value | count | ratio_percent |
| --- | --- | --- | --- |
| System_Failure | 0.0 | 6837 | 92.28 |
| System_Failure | 1.0 | 518 | 6.991 |
| System_Failure |  | 54 | 0.729 |
| ProtectiveStop | False | 7077 | 95.519 |
| ProtectiveStop | True | 278 | 3.752 |
| ProtectiveStop |  | 54 | 0.729 |
| GripLost | False | 7166 | 96.72 |
| GripLost | True | 243 | 3.28 |

## Timestamp 간격 통계

| metric | seconds |
| --- | --- |
| count | 7408.0 |
| mean | 3.55355467062635 |
| std | 218.58016180549012 |
| min | 0.984 |
| 25% | 1.003 |
| 50% | 1.005 |
| 75% | 1.007 |
| 90% | 1.01 |
| 95% | 1.013 |
| 99% | 1.016 |
| max | 18814.125 |

## 주요 Timestamp 간격

| diff_seconds_rounded | count |
| --- | --- |
| 1.005 | 1026 |
| 1.006 | 947 |
| 1.004 | 917 |
| 1.007 | 781 |
| 1.003 | 744 |
| 1.002 | 603 |
| 1.008 | 522 |
| 1.001 | 423 |
| 1.009 | 367 |
| 1.0 | 215 |
| 1.01 | 214 |
| 1.011 | 128 |
| 1.015 | 126 |
| 1.014 | 107 |
| 1.013 | 76 |
| 1.012 | 75 |
| 1.016 | 57 |
| 0.999 | 11 |
| 0.997 | 4 |
| 2.003 | 4 |

## 주의 사항

- `workload`, `movement speed`, `gripping force`는 논문에 설명된 공정 조건이지만, 공개 CSV의 직접 컬럼은 아니다.
- 과부하나 파지력 원인은 논문 기반 배경으로 설명하고, 본 데이터에서 직접 검증한 라벨처럼 쓰지 않는다.
- row 단위 random split은 시계열 로그의 일반화 성능을 낙관적으로 보일 수 있으므로 cycle 기준 split과 비교한다.
