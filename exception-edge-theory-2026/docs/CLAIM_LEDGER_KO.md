# 주장·증거·한계 원장

이 문서는 “무엇이 증명되었고, 무엇이 계산으로 관찰되었으며, 무엇이
아직 가설인가”를 한 표에서 추적하기 위한 공개 감사 원장이다.

## 1. 증명된 명제

| ID | 명제 | 적용 가정 | 근거 | 한계 |
|---|---|---|---|---|
| T1 | one-vertex relocation의 비용차는 \(s_e(w)-\delta_T(w)\)이며 \(M_T(e)\ge0\)은 그 이동족 비개선과 동치 | 고정 tour, 지정한 relocation | `THEORY_KO.md` §2, `PROOFS_KO.md` P1 | 전역 최적성이나 모든 k-opt를 보장하지 않음 |
| T2 | 임의의 고정 raw-\(\rho\) 임계값은 전역 포함 충분조건이 아님 | 일반 metric TSP | `PROOFS_KO.md` P2 | \(\rho\)의 통계 특징량 가치는 부정하지 않음 |
| T3 | 선택한 cut을 정확히 두 번 지나는 최적값은 내부 endpoint path와 교차 matching energy로 정확히 표현됨 | 고정 cut, 정확히 2회 교차 | `THEORY_KO.md` §3, `PROOFS_KO.md` P3 | 모든 cut과 4회 이상 교차를 자동 설명하지 않음 |
| T4 | 비-Hamiltonian \(G_0\)에서 \(r_{\rm top}=pc(G_0)\) | 완전 ambient graph, path cover 정의 | `THEORY_KO.md` §4, `PROOFS_KO.md` P4 | \(G_0\)가 Hamiltonian이면 예외 수 0 |
| T5 | \(z^*=\min_q Z_q^{=}\), \(Z_{\le q}\)는 비증가, 회수·잔여 에너지 분해 성립 | 기준그래프 상대 예외 수 | `THEORY_KO.md` §4, `PROOFS_KO.md` P5 | exact layer \(Z_q^{=}\) 자체는 비단조 |
| T6 | \(Z_1^{=}=\min_e(H_e+c(e))\), \(\kappa_e>1\iff G_e>0\) | \(c(e)>0\), finite \(Z_0,H_e\) | `THEORY_KO.md` §5, `PROOFS_KO.md` P6–P7 | 전역 포함·forcedness는 따라오지 않음 |
| T7 | cycle/path bound로 gain 안전구간과 exact \(q=1\) support 보존 후보 상한을 얻음 | 유효한 하한·상한 | `THEORY_KO.md` §6, `PROOFS_KO.md` P8–P9 | 구간이 넓으면 많은 pair가 미확정 |
| T8 | \(Z_0-P_H=(z^*-P_H)+(Z_0-z^*)\) | Euclidean hull perimeter 정의 | `THEORY_KO.md` §7, `PROOFS_KO.md` P10 | 내부 strain과 Delaunay frustration의 통계관계는 별도 가설 |

## 2. 동결 실험에서의 관찰

| ID | 관찰 | 선택·표본 | 검증 수준 | 반례·한계 |
|---|---|---|---|---|
| E0a | Stage 1 일반 기하 CUT은 Delaunay 밖 reference edge 11개 중 2개를 reference two-crossing interface에 귀속했고, 독립 two-crossing 인증은 0개였다 | TSPLIB strict-core 10, 공식 reference tour를 봉인 뒤 사용 | reference-only sealed pilot | forced-edge recall 아님; 9개 미귀속, raw artifact는 compact release 밖 |
| E0b | Stage 1B density CUT 92개 중 42개가 reference two-crossing이었지만, 전체 11개 reference exception과 잔여 9개의 density-CUT 귀속은 모두 0개였다 | 같은 10개 discovery 인스턴스 | exploratory sealed run | backbone 농축과 예외 생성 능력은 다름; 모집단 일반화 금지 |
| E1 | Stage 2 raw mandatory 178건 중 177건이 \(q=1\) | 합성 exact-small, targeted Hoey 포함 | Held–Karp exact spectrum | 일반 Euclidean 모집단 비율로 외삽 금지; \(q=2\) 사례 1건 존재 |
| E2 | natural core mandatory 51/51이 exact global·exclusive \(q=1\) | uniform 300 + structured mixture 300 | exact DP, 동결 holdout | \(n\le12\) 합성 표본 |
| E3 | 위 51건 중 48건이 현재 1-tree 구간으로 안전 인증됨 | 같은 natural core | 충분 인증, 위반 0 | ring에서는 7/26으로 bound coverage가 낮음 |
| E4 | natural core pair 15,054→644로 축소하며 exact support 600개 보존 | exact-small pair pool | exact support audit | 대규모 계산량·분포로 바로 외삽 금지 |
| E5 | LIN318 추가 후보 512개 중 13개가 safe lower positive | 988+512 봉인 후보영역 | 512/512 feasible witness 독립 검증 | exact \(\kappa\)·recall 아님; 499개는 미확정 |
| E6 | closure seed와 순차 개입으로 42,210→42,118→42,108 | LIN318 한 인스턴스, 전수 scan 후 탐색 | deterministic reproduction | LKH seed 사용, holdout·동일예산 대조 없음, 최적 미도달 |

관찰 E0a–E6은 정리 T1–T8로 승격하지 않는다.

## 3. 다음 검증 대상 가설

| ID | 가설 | 필요한 검증 |
|---|---|---|
| H1 | shell pressure와 endpoint accessibility가 mandatory/\(q=1\) 발생을 사전 예측한다 | 새 좌표분포·새 규모 holdout, calibration·PR-AUC |
| H2 | strain 집중도가 \(q^*=1\)과 \(q^*\ge2\)를 구분한다 | exact-small의 사전 동결 분류 실험 |
| H3 | 안전 \(\kappa\)-positive edge가 다른 기하 점수보다 basin portal을 잘 순위화한다 | 동일 호출수·시간 예산의 nearest/stretch/random 대조 |
| H4 | LIN318 잔여는 개별 portal보다 2/3-edge bundle 호환성과 순서에 크게 좌우된다 | 사전 동결 bundle 후보·순서·lock ablation |
| H5 | CUT의 강한 two-crossing 인증과 closure endpoint pair 사이에 재사용 가능한 충분조건이 있다 | exact-small 전수 cut–closure 대응지도 |

## 4. 명시적 비주장

현재 자료는 다음을 지지하지 않는다.

1. TSP의 다항시간 exact 해법 또는 NP-hardness 해소
2. 모든 Euclidean TSP 예외간선의 완전한 기하·위상 분류
3. 보편적인 양의 예외간선 비율 하한
4. \(\kappa_e>1\)인 간선의 전역 최적투어 포함 또는 forcedness
5. LIN318 1,500개 후보를 임계점 이론이 처음부터 생성했다는 주장
6. 13/512를 최적간선 recall 또는 exact \(\kappa\) 정확도라고 부르는 것
7. 나머지 499개를 음성이라고 부르는 것
8. 42,118 또는 42,108을 LKH-free 독립 solver 성능이라고 부르는 것
9. 한 LIN318 개입 간선이 비용 개선의 전부를 인과적으로 만들었다는 주장
10. 기존 LKH, Concorde 또는 최신 NCO 방법보다 우월하다는 주장

## 5. 변경 규칙

새 결과를 추가할 때는 다음 네 항목을 함께 기록한다.

- 표본과 선택 시점
- ground truth 접근 시점
- exact / 안전 하한 / 탐색적 중 어느 수준인지
- 반례, 미확정, 실패 결과

수치가 좋아져도 이 분류를 생략하지 않는다.
