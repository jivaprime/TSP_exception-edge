# 실험 설계와 데이터 누설 방지

## 1. 실험의 세 층

### 1.1 이론·구현 단위검증

작은 그래프에서 brute force 또는 독립 Held–Karp 계산과 비교한다.

- exact exception-count spectrum
- endpoint Hamiltonian path
- pair-closure identity
- cycle/path 1-tree 하한의 안전성
- gain sandwich와 support 보존
- SEC-MILP subtour elimination
- atomic 2/3-opt와 lock 보존

이 층은 정리의 구현 오류를 찾는 역할을 한다.

### 1.2 exact-small corpus

\(n\le12\) 합성 좌표에서 모든 관련 optimum을 exact하게 계산한다.
주요 family는 strict-convex, uniform, structured mixture,
ring-interior, corridor, Hoey fixture와 jitter다.

각 좌표에 대해 raw Euclidean과 TSPLIB `EUC_2D` 목적함수를 분리한다.
기록량은 다음과 같다.

- \(Z_q^{=}\)와 \(Q^*\)
- mandatory-exception 여부
- \(q=1\) global/exclusive 여부
- pair별 exact \(H_e,C_e,G_e,\kappa_e\)
- cycle/path 1-tree 구간
- 안전 양성·안전 음성·미확정
- support-preserving 후보집합

targeted Hoey family는 자연 발생률 추정에 섞지 않는다. natural core는
uniform과 structured mixture를 합친 동결 600개로 보고한다.

### 1.3 LIN318 규모 확장

LIN318에서는 모든 \(H_e\)를 exact하게 풀지 않았다. 실험은 다음
순서로 분리했다.

1. **봉인 후보영역:** weak-Delaunay 988개와 별도 후보 512개를 합친
   1,500개 그래프
2. **기준값 인증:** weak-Delaunay 제한 TSP의 SEC-MILP 하한과
   검증된 feasible tour 상한이 42,231에서 일치
3. **forced-closure scan:** 512개 각각에 대해 LKH가
   `weak-Delaunay + e` 안에서 \(e\)를 포함하는 cycle을 탐색
4. **독립 검증:** 318정점 permutation, forced edge, 나머지 간선의
   weak-Delaunay 소속, `EUC_2D` 비용을 재계산
5. **안전 하한:** \(42{,}231-\overline H_e-c(e)>0\)인 pair만 양성
6. **basin 실험:** closure tour를 1,500-edge strict 2/3-opt 시드로
   사용하고, atomic insertion의 added-set을 잠근 뒤 풀어 재하강

## 2. ground-truth 격리

후보 선택, closure scan, 42,210→42,118→42,108의 탐색에는 알려진
최적투어나 최적값을 입력하지 않았다. 알려진 최적값 42,029는 결과
동결 뒤 gap을 계산하는 사후 평가에만 사용했다.

1,500개 후보 안에 알려진 optimum witness의 간선이 모두 들어간다는
사실도 사후 감사다. 이는 어떤 optimum의 존재 witness이지, 모든
최적투어 포함이나 후보 생성의 선견성을 증명하지 않는다.

## 3. LKH의 정확한 역할

LKH는 LIN318 규모에서 다음 두 역할을 했다.

- forced edge를 포함하는 feasible closure witness 생성
- 그 closure tour를 basin 탐색의 전역 시드로 제공

LKH 출력은 그대로 믿지 않고 별도 코드로 구조와 비용을 재검사했다.
그러나 42,118은 LKH tour에서 출발했으므로 LKH-free 결과가 아니다.
LKH는 exact \(H_e\) oracle도 아니며, witness를 찾지 못했다고 음성을
말할 수도 없다.

## 4. 후보 1,500개의 출처

완전그래프 간선은 50,403개다.

| 구성 | 간선 수 |
|---|---:|
| weak-Delaunay | 988 |
| 별도 추가 후보 | 512 |
| 합계 | 1,500 |
| 완전그래프 대비 | 2.976% |

1,500개는 \(\kappa\)가 완전그래프에서 직접 생성한 후보집합이 아니다.
임계점 검사는 별도로 만들어진 512개 추가 후보를 screening한다.

공개된 Colab artifact에서 `static_local_b512`와
`static_nearest_b512`의 후보 CSV는 동일한 SHA-256
`aa75561a...d4a69`를 갖고 byte 단위로 같다. 이 사실을 대조군
독립성처럼 표현하지 않는다.

## 5. local-search 개입

출발점 \(T_0=42{,}210\)은 봉인된 2-factor를 candidate-only로
결정론적으로 patch한 뒤 strict best-improvement 2-opt/3-opt를
적용해 재구성한다.

목표 edge를 현재 tour에 넣는 최소 즉시 장벽 atomic move를 찾고,
그 move가 추가한 간선집합 전체를 잠근다.

1. atomic insertion
2. added-set lock 아래 strict 2/3-opt
3. lock 해제
4. strict 2/3-opt 재하강

간선 하나만 잠그지 않고 atomic move 전체를 잠그는 이유는 첫
하강에서 perturbation이 즉시 취소되는 것을 막고, 이동 단위를
불가분하게 유지하기 위해서다.

## 6. 음성 결과도 보존

- Colab zero-base pilot의 15조건은 60초 안에 모두
  `no_cycle_found`였다.
- \(T_0\)에 없는 안전 양성 10개를 각각 직접 삽입했지만 모두 lock
  해제 후 42,210으로 돌아왔다.
- 초기 safe lower가 비양성인 상위 32개를 더 강한 LKH 예산으로
  재탐색해도 새 안전 양성은 없었다.
- 42,108에서 그 32개를 safe-positive anchor와 함께 개입해도 개선은
  없었다.

이 결과들은 “그 간선들이 무효”라는 증명이 아니다. 현재 witness
예산과 개입 연산자에서 개선을 찾지 못했다는 결과다.

## 7. 아직 없는 대조

다음 비교는 아직 완료되지 않았다.

- safe-threshold 대 local-stretch / nearest / random의 동일 호출수
- 동일 wall-clock과 LKH 호출 예산
- 사전 동결한 bundle 후보와 순서
- 여러 대형 인스턴스 holdout
- closure seed 없이 threshold score만 쓰는 독립 solver

따라서 LIN318 결과를 solver 순위나 인과 효과로 보고하지 않는다.
