# Exception-Edge Theory Guided PPO TSP Solver

## 🚀 성능 하이라이트 (Zero-shot, No Pretraining)

| Instance | Nodes | Best PPO Score (Ours) | Best Known (Public/Observed) | Gap (%) | Notes |
|----------|-------|----------------------|------------------------------|---------|-------|
| d1291 | 1,291 | **51,646** | 51,827 *(observed, non-exhaustive)* | **-0.35%** | PPO-only, on-instance |
| gr431 | 431 | **171,702** | 171,414 *(optimal)* | +0.17% | PPO-only, on-instance |

> **중요 공지 (정확성/검증 관련)**
> 
> "Best Known (Public/Observed)"는 공개 자료와 제한된 탐색으로 **우리가 확인한 범위 내** 최선 기록입니다.
> **엄밀한 전수조사나 공식 인증(SOTA 확정)을 보장하지 않습니다.**
> 
> 만약 **공식적으로 문서화되어 있거나 재현 가능한 더 좋은 ML 기록**이 있다면, 
> 이슈/PR로 **근거 링크 + 로그 + 재현 절차**를 함께 공유해 주세요.

> **Zero-shot On-instance RL**
> 
> 본 솔버는 **타깃 인스턴스 단독 롤아웃만으로 정책을 처음부터 학습**합니다.
> 외부 데이터셋, 합성 데이터, 사전학습(pretraining)을 사용하지 않습니다.

---

## 1. 예외간선(Exception Edge) 이론 요약

유클리드/비균질 TSP에서 대부분의 간선은 "로컬 메쉬(근접 그래프)"에 포획되지만, **소수의 예외 간선**이 최적해의 난이도를 결정합니다. 본 프로젝트는 예외 간선을 **기하/위상 구조로 분해**하고, 그 구조를 **탐색 정책(PPO)**에 직접 주입합니다.

### 1.1 효과적 지지집합 (Effective Support, Land)

밀도 함수 λ(x)가 주어졌을 때, 임계값 τ로 유효 지지집합을 정의합니다:

```
D_τ := {x ∈ D : λ(x) ≥ τ}
```

직관적으로 D_τ는 "도시가 실제로 분포한 땅(Land)"이며, D\D_τ는 "희박/공허한 영역(Sea/Void)"에 해당합니다.

### 1.2 우회비율(Detour Ratio)과 타원(2-hop Ellipse)

점 u, v에 대해 제3점 w를 경유한 우회비:

```
ρ(u,v) := min_{w ≠ u,v} [d(u,w) + d(w,v)] / d(u,v)
```

또한 η > 0에 대해 "우회가 (1+η) 이하인 점들의 집합"인 타원을 정의합니다:

```
E(u,v; η) := {x : d(u,x) + d(x,v) ≤ (1+η)·d(u,v)}
```

ρ(u,v)가 크다는 것은 "가까운 우회로가 거의 없다"는 뜻이고, 이는 예외 간선 후보를 강하게 시사합니다.

### 1.3 타원 질량(Ellipse Mass)

본 이론의 중심 지표는 **타원이 Land에서 확보하는 질량**입니다:

```
M_τ(u,v; η) := n · ∫_{E(u,v;η) ∩ D_τ} λ(x) dx
```

| M_τ 크기 | 의미 | 예외성 |
|---------|------|--------|
| **큼** | 타원 내부에 점이 많음 → 우회로 존재 확률 높음 | 낮음 (Bulk) |
| **작음** | 타원이 Sea를 가로지름 → 우회로 거의 없음 | 높음 (Bridge) |

#### 구현에서의 근사

연속 적분은 코드에서 **타원 내부 점 개수(count)** 기반으로 근사합니다:

```python
# k-NN 후보군에서 타원 내부에 들어오는 점 수를 카운트
mass = count(w for w in candidates if d(u,w) + d(w,v) <= (1+η)*d(u,v))
```

### 1.4 A+B 조건: 두께(Thickness) → 질량 하한

비균질에서 "No-go"를 깨뜨리는 핵심은 M_τ가 O(1)로 내려앉는 병목/회랑 구조입니다. 이를 배제하기 위해:

- **(A) 두께(Thickness)**: relevant ellipse가 D_τ 안에서 일정 비율의 면적을 확보
- **(B) 질량 하한(Mass Lower Bound)**: M_τ ≳ c·log(n)

(A) → (B)로 연결되면, union bound 기반 희귀성 제어가 다시 살아납니다.

### 1.5 v4.1 개선: Length-gated Mass Condition

v4.1에서는 mass 조건을 **긴 간선에서만** 강하게 적용합니다:

```python
def is_exception_edge(rho, mass, norm_length):
    # Micro-bridge: rho가 크면 길이와 무관하게 예외
    if rho > rho_hi:
        return True
    
    # Macro-bridge: "긴 간선"에서만 mass 위반을 예외로
    if norm_length >= long_threshold and mass < mass_threshold:
        return True
    
    return False
```

**효과**: bulk 간선에서 mass 신호 오염 제거 → 진짜 예외간선만 집중

---

## 2. 아키텍처: Triple Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                        MANAGER                               │
│  관찰: gap, stagnation, thickness (λ₂), 시간, 예외비율        │
│  행동: EXPLORE | STANDARD | MACRO_FOCUS | ESCAPE | FINALIZE  │
│  학습: PPO (sparse reward - 구간 개선률)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │ mode 선택
┌─────────────────────────▼───────────────────────────────────┐
│                         SCOUT                                │
│  밀도 클러스터 간 macro-bridge 후보 생성                       │
│  점수: ρ (우회비), mass (희귀성), normalized length           │
│  학습: Self-Imitation Learning (elite tour edges)           │
└─────────────────────────┬───────────────────────────────────┘
                          │ 후보 간선
┌─────────────────────────▼───────────────────────────────────┐
│                        WORKER                                │
│  예외 간선 중심으로 3-opt move 제안                           │
│  PPO 정책으로 move 선택                                      │
│  학습: PPO (immediate delta reward)                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 이론 레이어 (Feature / Gating)

- 입력 좌표로부터 D_τ, ρ, M_τ 근사치, 경계 레이어/병목 신호 계산
- 간선 후보를 **Bulk / Boundary / Micro / Macro** 성격으로 분류

### 2.2 PPO 레이어 (Smart 3-opt 정책)

- **상태**: 이론 레이어 신호 (ρ, M_τ proxy, boundary proximity, stagnation 등)
- **행동**: 어떤 3-opt 재연결을 시도할지
- **보상**: 투어 길이 감소 + 정체 탈출 유도

### 2.3 로컬 최적화 레이어

- **2-opt**: 빠른 정리 단계 (교차 제거 + 초기 개선)
- **3-opt**: PPO가 선택한 move 실행

---

## 3. Macro 예외 처리 전략

| 예외 유형 | 제어 방식 | 특징 |
|----------|----------|------|
| **Micro** | 희귀성 (M_τ ≳ c·log n) | 확률적으로 통제 가능 |
| **Macro** | 구조적 강제 | 병목/분리에 의해 필수적으로 발생 |

Macro 예외는 희귀성만으로는 막을 수 없어서:

1. **탐지**: M_τ가 작거나 병목 지표가 낮을 때 Macro 가능성 상승
2. **처리**: Scout가 Macro 후보를 명시적으로 생성 → PPO가 직접 다룸
3. **v4.1 개선**: Scout endpoint를 active node에 보너스 → 투어에 없는 macro-bridge도 탐색

---

## 4. 스케일링 법칙 (Scaling Laws)

이론에서 도출된 파라미터 스케일링:

```python
mass_threshold = c · log(n)           # 타원 질량 임계값
core_ratio = c · n^(-1/3)             # 예외 간선 코어 비율  
long_threshold = c · √log(n)          # 긴 간선 판정 (v4.1)
max_active_nodes = 2 · n · core_ratio # 집중 탐색 노드 수
```

| n | mass_threshold | core_ratio | max_active_nodes |
|---|----------------|------------|------------------|
| 431 | ~9.1 | ~0.26 | ~226 |
| 1,291 | ~10.7 | ~0.18 | ~474 |
| 5,000 | ~12.8 | ~0.12 | ~1,168 |

---

## 5. 실행 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA A100 |
| 플랫폼 | Google Colab |
| 학습 방식 | On-instance PPO (from scratch) |
| 사전학습 | 없음 |
| 외부 데이터 | 없음 |

### 재현성 체크리스트

- [x] Seed 고정 (random, numpy, torch)
- [x] Numba 캐시 (`@njit(cache=True)`)
- [x] 버전 명시 (`__version__ = "4.1.0"`)
- [x] CLI 파라미터 조절 가능
- [x] 로그 저장 (iteration, time, best, mode)

---

## 6. 사용 방법

### 설치

```bash
pip install numpy scipy torch numba matplotlib
```

### CLI 실행

```bash
# 기본 실행
python tsp_solver.py --input d1291.tsp --time 3000 --verbose

# 시드 지정
python tsp_solver.py --input d1291.tsp --time 3000 --seed 42

# 결과 저장
python tsp_solver.py --input d1291.tsp --time 3000 --output solution.txt
```

### Python API

```python
from tsp_solver import solve, parse_tsplib, SolverConfig

# 문제 로드
coords, edge_type, meta = parse_tsplib("d1291.tsp")

# 설정
cfg = SolverConfig(
    time_total=3000.0,
    seed=42,
    verbose=True
)

# 풀이
result = solve(coords, edge_type=edge_type, cfg=cfg)

print(f"Best tour length: {result['length']}")
```

---

## 7. 파일 구조

```
exception-edge-tsp/
├── README.md              # 이 문서
├── tsp_solver.py          # 메인 솔버 (단일 파일)
├── requirements.txt       # 의존성
├── logs/
│   └── d1291_ppo_log.txt  # 실험 로그
└── data/
    ├── d1291.tsp          # 테스트 인스턴스
    └── gr431.tsp
```

---

## 8. 라이선스

- **코드**: MIT License
- **데이터**: TSPLIB/공개 인스턴스는 각 출처의 정책을 따릅니다.

---

## 9. 인용

```bibtex
@software{exception_edge_tsp,
  title={Exception-Edge Theory Guided PPO TSP Solver},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/exception-edge-tsp}
}
```

---

## 10. 핵심 기여

1. **이론적 프레임워크**: 예외 간선(Exception Edge)을 기하/위상적으로 분류
2. **스케일링 법칙**: mass ~ c·log(n), core ~ c·n^(-1/3) 도출 및 검증
3. **Zero-shot 학습**: 사전학습 없이 단일 인스턴스에서 경쟁력 있는 성능
4. **해석 가능성**: "왜 이 간선이 중요한가"에 대한 명시적 답변 제공
