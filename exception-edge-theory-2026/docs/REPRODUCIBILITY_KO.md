# 재현성과 공개 artifact 안내

## 1. 권장 환경

- Python 3.12 이상
- NumPy 2.0 이상
- SciPy 1.14 이상

검증에 사용한 Windows/Python 3.13 top-level 버전은
`requirements-lock.txt`에 기록했다. lock은 top-level 패키지 버전을
고정하지만 운영체제별 wheel의 byte hash까지 고정한 것은 아니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# Linux/macOS
.venv/bin/python -m pip install -r requirements.txt
```

Colab에서는 별도 `venv`를 만들 필요가 없다. 런타임 Python에 직접
의존성을 설치하고 이 폴더를 작업 디렉터리로 두는 편이 단순하다.

## 2. 1분 감사

```bash
python verify_public_results.py
```

기본 감사가 확인하는 항목:

1. weak 후보 988개, target 후보 1,500개, 차집합 512개
2. target CSV의 봉인 SHA-256
3. `static_local_b512`와 `static_nearest_b512` snapshot의 동일성
4. 봉인 2-factor에서 \(T_0=42{,}210\) 재구성
5. 512행의 closure/path/gain/\(\kappa\) 산술
6. verified witness 512, safe lower positive 13
7. 공개 tour 네 개의 Hamiltonicity, candidate membership, 비용
8. weak-Delaunay 제한 하한·상한 42,231의 일치
9. optimum tour를 읽지 않았다는 실행 계약

## 3. 전체 basin 재현

```bash
python verify_public_results.py --full
```

추가로 공개 bundle과 45개 retained witness만 이용해 다음을 다시
계산한다.

\[
42{,}210\to42{,}118\to42{,}108.
\]

재현은 다음 45개 witness를 필요로 한다.

- safe-positive 13개
- initial-scan 비양성·미확정 중 closure upper 기준 top 32

512개 per-edge 작업 디렉터리를 모두 포함하지 않아도 runner가 실제
조회하는 witness는 이 45개다.

## 4. 테스트

```bash
python -m unittest discover -s tests -v
```

테스트 범위:

- exact closure spectrum 대 brute force
- path-cover 위상 예외 수
- pair-closure identity
- safe gain interval과 support 보존
- LKH task 작성과 독립 witness verifier
- resume 가능한 병렬 scan
- atomic 2-opt/3-opt, added-set lock, bundle invariant
- SEC-MILP subtour elimination
- compact public artifact 감사
- LIN318 end-to-end basin 재현

LKH 호출 테스트는 mock tour를 사용한다. 공개 test suite 자체는 LKH
binary를 요구하지 않는다.

## 5. 공개 폴더 구조

```text
exception-edge-theory-2026/
├── README.md
├── README_KO.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── requirements-lock.txt
├── verify_public_results.py
├── run_lin318_threshold_basin.py
├── docs/
├── exception_edge/
├── tests/
├── data/
│   ├── lin318_reproduction_inputs.zip
│   └── lin318_reproduction_manifest.json
└── results/
    ├── exact_small/
    └── lin318/
```

## 6. compact LIN318 bundle

`data/lin318_reproduction_inputs.zip`은 원래 Colab 결과 ZIP 전체를
재배포하지 않는다. 다음 항목만 원래 내부 경로를 유지해 담았다.

1. weak 후보 CSV와 seal
2. static-local-b512 후보 CSV와 seal
3. static-nearest-b512 후보 CSV와 seal
4. static-shortest-b512의 봉인된 2-factor MILP audit JSON

nearest snapshot 두 항목은 재현에 필수는 아니지만 local snapshot과
동일했다는 사실을 공개 감사하기 위해 포함했다.

## 7. raw scan의 재실행

512개 forced closure를 LKH로 처음부터 다시 만들려면 공개 폴더 외에
다음이 필요하다.

- TSPLIB LIN318 원본
- 사용자가 별도로 설치한 LKH
- weak-Delaunay 및 target candidate snapshot

실행 인터페이스는:

```bash
python -m exception_edge.lin318_threshold_escape \
  --help
```

LKH와 TSPLIB 원본은 각 배포정책 때문에 이 저장소에 포함하지 않는다.
공개 결과의 hash와 독립 verifier로 기존 witness의 무결성을 확인할
수 있다.

## 8. 제한값 42,231의 감사

`results/lin318/restricted_baseline/`에는 다음을 포함한다.

- `restricted_bounds.csv`: lower/upper 일치표
- `restricted_milp_rounds.csv`: SEC 반복별 component와 cut 수
- `restricted_milp_audit.json`: 최종 변수·간선·검증 정보
- `lin318_weak_delaunay_witness.tour`: feasible upper witness

MILP는 degree-2 model에 subtour elimination constraints를 반복
추가했다. solver가 보고한 lower bound는 정수 비용 문제의 유효한
하한으로 감사했고, 별도 tour verifier의 상한과 42,231에서 일치한다.

## 9. 결과 경로

공개된 JSON·CSV는 개인 로컬 절대경로를 제거하거나 처음부터
경로중립으로 만들었다. 재실행 중 생성되는 출력은 사용자가 지정한
output 경로를 기록할 수 있으나, 그런 실행별 파일은 Git에 자동으로
추가되지 않는다.

## 10. 완전한 원자료와 공개본의 차이

제외된 원자료:

- 38MB pair-level table
- 512개 LKH stdout/stderr/parameter/work directory
- 여러 quick/revised/custom 중복 실행
- 10MB Colab 전체 archive
- 수정 중이던 notebook

공개본은 결론을 다시 계산하고 핵심 witness를 감사하는 데 필요한
최소 자료를 보존한다. 향후 raw 전체를 공개할 경우 Git tree가 아니라
버전·schema·SHA-256을 붙인 별도 archival release를 사용한다.
