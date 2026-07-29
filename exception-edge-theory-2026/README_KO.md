# 예외간선 이론 2026: 엄밀 이론·테스트·결과 공개본

[이론 본문](docs/THEORY_KO.md) ·
[증명 해설](docs/PROOFS_KO.md) ·
[실험 설계](docs/EXPERIMENTS_KO.md) ·
[결과와 해석](docs/RESULTS_KO.md) ·
[주장 원장](docs/CLAIM_LEDGER_KO.md) ·
[재현 안내](docs/REPRODUCIBILITY_KO.md)

이 폴더는 기존 저장소에 흩어진 실험을 그대로 복사한 것이 아니라,
외부 검토에 필요한 내용만 허용목록 방식으로 선별한 공개 패키지입니다.
구성의 중심은 다음 세 가지입니다.

1. 예외간선 이론의 전체 구조와 증명·가설의 경계
2. exact-small 계산, 안전 인증, LIN318 실험에 사용한 코드와 테스트
3. 동결된 결과표·대표 투어·해시와 과장 없이 쓴 해석

## 한 문장 요약

예외간선은 간선 자체의 절대 속성이 아니라 기준 후보그래프 \(G_0\)에
상대적인 개념이며, 이 연구는

\[
\text{국소 이상신호}
\to \text{CUT 인터페이스}
\to q\text{-closure spectrum}
\to \kappa=1\text{ 임계점}
\to \text{안전 인증}
\to \text{basin 개입}
\]

이라는 계층을 세웠습니다.

비기준 간선 \(e=\{s,t\}\)를 하나 허용했을 때, 기준 간선만 쓰는 최저
Hamiltonian \(s\)-\(t\) 경로 비용을 \(H_e\), 기준 cycle 최적값을
\(Z_0\)라 하면

\[
\kappa_e=\frac{Z_0-H_e}{c(e)}>1
\iff H_e+c(e)<Z_0.
\]

이는 \(e\)가 포함된 **한 개 예외간선 closure**가 기준 cycle을 이긴다는
정확한 조건입니다. \(e\)가 전역 최적투어에 반드시 포함된다는 조건은
아닙니다.

## 결과 카드

| 단계 | 핵심 결과 | 증거 수준 |
|---|---:|---|
| Stage 2 | raw mandatory 178건 중 177건이 \(q=1\) | exact-small 관찰 |
| Stage 3 natural core | mandatory 51/51이 exact global·exclusive \(q=1\) | 동결 표본 내 exact |
| Stage 3 안전 인증 | 48/51 mandatory, 88/98 beneficial edge | 1-tree 기반 충분 인증 |
| Stage 3 후보 상한 | 15,054 → 644, exact support 600개 보존 | 동결 표본 내 감사 |
| LIN318 후보 그래프 | 50,403개 중 1,500개(2.976%) | 사전 봉인된 탐색 영역 |
| LIN318 임계점 스캔 | 추가 후보 512개 중 안전 양성 13개 | 하한 양성; 499개 미확정 |
| LIN318 basin 탐색 | 42,210 → 42,118 → 42,108 | LKH closure 시드 포함 탐색 |
| 알려진 최적값과 차이 | 79, 0.18797% | 사후 비교 |

## LIN318에서 반드시 구분할 사실

- 1,500개는 임계점 공식이 완전그래프에서 직접 뽑은 집합이 아닙니다.
  weak-Delaunay 988개와 별도 후보 512개의 합입니다.
- 공개 artifact에서 `static_local_b512`와 `static_nearest_b512`의 후보
  스냅숏은 동일합니다. 공개 감사 코드가 byte 단위 동일성을 확인합니다.
- Colab 제로베이스 pilot의 15개 조건은 각 60초 안에 모두
  `no_cycle_found`였습니다. 이것은 solver 성공 결과가 아닙니다.
- 13/512는 exact \(\kappa\) recall이 아니라, 유효한
  \(Z_{\mathrm{lower}}=42{,}231\)과 feasible closure witness로 얻은
  안전 양성 수입니다. 나머지 499개는 음성이 아닙니다.
- 42,118은 LKH가 만든 forced-closure tour를 시드로 썼습니다.
  42,108은 전수 스캔 뒤 수행한 순차 개입입니다. 둘 다 LKH-free
  독립 솔버나 사전 동결 holdout 성능으로 해석하지 않습니다.
- 최적값 42,029에는 도달하지 못했습니다.

## 빠른 검증

```bash
python -m pip install -r requirements.txt
python verify_public_results.py
python -m unittest discover -s tests -v
```

45개 공개 witness까지 이용해 `42,210 → 42,118 → 42,108`을 다시
계산하려면 다음을 실행합니다.

```bash
python verify_public_results.py --full
```

기본 감사와 전체 재현은 알려진 최적투어를 입력으로 읽지 않습니다.

## 공개 범위

포함한 것:

- 통합 이론과 핵심 정리의 증명 해설
- exact closure·1-tree 안전구간·SEC-MILP·LIN318 개입 코드
- 30개 공개 테스트(환경에 따라 테스트 수는 이후 늘어날 수 있음)
- exact-small 요약표와 그림
- LIN318 512행 스캔표, 재현에 필요한 45개 witness, 대표 투어
- weak-Delaunay 제한값 42,231의 하한·상한 일치 근거

제외한 것:

- 중복된 `quick`, `revised`, `custom`, 임시 결과
- 38MB pair 원자료와 재생성 가능한 수백 개 작업 디렉터리
- 개인 로컬 경로가 든 원본 결과 파일
- 수정 중이던 Colab 노트북
- LKH 실행파일과 TSPLIB 원본
- 다른 연구 주제와 개인 분석 자료

Stage 2·3의 공개 코드는 core oracle과 안전구간·테스트에 집중한다.
541/600개 exact-small corpus를 처음부터 생성한 study
driver·generator·config와 대형 pair 원자료는 포함하지 않으며, 공개
감사는 동결 summary와 core 계산을 검증한다. 앞선 Stage 1 CUT 및 Stage
1B density-CUT의 음성 결과는 [결과 문서](docs/RESULTS_KO.md)에
요약하되, 그 원자료와 실행 코드는 compact release 범위에서 제외했다.

## 이 연구가 주장하지 않는 것

TSP의 다항시간 완전해결, NP-hardness의 해소, 모든 예외간선의 완전한
분류, LKH보다 우월한 solver, LIN318에서 특정 한 간선의 인과 효과,
또는 보편적인 \(\kappa\) 분포법칙을 주장하지 않습니다.

현재 가장 강한 결론은 더 구체적입니다. 예외간선 현상의 큰 부분이
무작위 잔차라기보다 Hamiltonian path를 닫는 구조적 closure로
관찰되며, 그중 \(q=1\) 층은 정확한 임계식과 안전한 계산 구간으로
다룰 수 있습니다. LIN318의 남은 차이는 단일 portal의 존재보다 여러
portal·companion 간선의 호환성과 순서 문제를 다음 과제로 가리킵니다.
