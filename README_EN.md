# Exception-Edge Theory Guided PPO TSP Solver

## Performance Highlights (Zero-shot, No Pretraining)

| Instance | Nodes | Best PPO Score (Ours) | Best Known (Public/Observed) | Gap (%) | Notes |
|----------|-------|----------------------|------------------------------|---------|-------|
| d1291 | 1,291 | **51,646** | 51,827 *(observed, non-exhaustive)* | **-0.35%** | PPO-only, on-instance |

> **Important Notice (Accuracy/Verification)**
> 
> "Best Known (Public/Observed)" represents the best records **we have identified** through public sources and limited search.
> **We do not guarantee exhaustive verification or official SOTA certification.**
> 
> If you have **officially documented or reproducible better ML records**, 
> please share via Issue/PR with **source links + logs + reproduction steps**.

> **Zero-shot On-instance RL**
> 
> This solver **learns policies from scratch using only rollouts on the target instance**.
> No external datasets, synthetic data, or pretraining is used.

---

## 1. Exception Edge Theory Summary

In Euclidean/non-homogeneous TSP, most edges are captured by the "local mesh" (proximity graph), but **a small number of exception edges** determine the difficulty of finding optimal solutions. This project **decomposes exception edges by geometric/topological structure** and directly injects this structure into the **search policy (PPO)**.

### 1.1 Effective Support (Land)

Given a density function λ(x), we define the effective support with threshold τ:

```
D_τ := {x ∈ D : λ(x) ≥ τ}
```

Intuitively, D_τ represents "land where cities are actually distributed," while D\D_τ represents "sparse/void regions (Sea/Void)."

### 1.2 Detour Ratio and 2-hop Ellipse

For points u, v, the detour ratio via a third point w:

```
ρ(u,v) := min_{w ≠ u,v} [d(u,w) + d(w,v)] / d(u,v)
```

For η > 0, we define the ellipse of "points with detour at most (1+η)":

```
E(u,v; η) := {x : d(u,x) + d(x,v) ≤ (1+η)·d(u,v)}
```

A large ρ(u,v) indicates "almost no nearby detours exist," strongly suggesting an exception edge candidate.

### 1.3 Ellipse Mass

The central metric of this theory is **the mass the ellipse captures from Land**:

```
M_τ(u,v; η) := n · ∫_{E(u,v;η) ∩ D_τ} λ(x) dx
```

| M_τ Size | Meaning | Exception Level |
|----------|---------|-----------------|
| **Large** | Many points inside ellipse → high probability of detour existence | Low (Bulk) |
| **Small** | Ellipse crosses Sea → almost no detours | High (Bridge) |

#### Implementation Approximation

In code, continuous integration is approximated via **counting points inside the ellipse**:

```python
# Count points in local k-NN that fall within the ellipse
mass = count(w for w in candidates if d(u,w) + d(w,v) <= (1+η)*d(u,v))
```

### 1.4 A+B Condition: Thickness → Mass Lower Bound

The key to breaking "No-go" in non-homogeneous settings is bottleneck/corridor structures where M_τ drops to O(1). To exclude this:

- **(A) Thickness**: Relevant ellipse secures a certain fraction of area within D_τ
- **(B) Mass Lower Bound**: M_τ ≳ c·log(n)

When (A) → (B) holds, union bound-based rarity control is restored.

### 1.5 v4.1 Improvement: Length-gated Mass Condition

In v4.1, the mass condition is applied **only to long edges**:

```python
def is_exception_edge(rho, mass, norm_length):
    # Micro-bridge: exception if rho is large regardless of length
    if rho > rho_hi:
        return True
    
    # Macro-bridge: mass violation counts as exception only for "long edges"
    if norm_length >= long_threshold and mass < mass_threshold:
        return True
    
    return False
```

**Effect**: Removes mass signal contamination from bulk edges → focuses only on true exception edges

---

## 2. Architecture: Triple Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                        MANAGER                               │
│  Observes: gap, stagnation, thickness (λ₂), time, exc_ratio │
│  Actions: EXPLORE | STANDARD | MACRO_FOCUS | ESCAPE | FINALIZE│
│  Learning: PPO (sparse reward - interval improvement rate)   │
└─────────────────────────┬───────────────────────────────────┘
                          │ mode selection
┌─────────────────────────▼───────────────────────────────────┐
│                         SCOUT                                │
│  Generates macro-bridge candidates between density clusters  │
│  Scores: ρ (detour ratio), mass (rarity), normalized length │
│  Learning: Self-Imitation Learning (elite tour edges)       │
└─────────────────────────┬───────────────────────────────────┘
                          │ candidate edges
┌─────────────────────────▼───────────────────────────────────┐
│                        WORKER                                │
│  Proposes 3-opt moves centered on exception edges           │
│  PPO policy selects moves                                   │
│  Learning: PPO (immediate delta reward)                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Theory Layer (Feature / Gating)

- Computes D_τ, ρ, M_τ approximations, boundary layer/bottleneck signals from input coordinates
- Classifies edge candidates as **Bulk / Boundary / Micro / Macro**

### 2.2 PPO Layer (Smart 3-opt Policy)

- **State**: Theory layer signals (ρ, M_τ proxy, boundary proximity, stagnation, etc.)
- **Action**: Which 3-opt reconnection to attempt
- **Reward**: Tour length reduction + stagnation escape incentive

### 2.3 Local Optimization Layer

- **2-opt**: Quick cleanup phase (crossing removal + initial improvement)
- **3-opt**: Executes moves selected by PPO

---

## 3. Macro Exception Handling Strategy

| Exception Type | Control Method | Characteristics |
|----------------|----------------|-----------------|
| **Micro** | Rarity (M_τ ≳ c·log n) | Probabilistically controllable |
| **Macro** | Structurally forced | Necessarily occurs due to bottlenecks/separations |

Macro exceptions cannot be prevented by rarity alone:

1. **Detection**: Macro likelihood increases when M_τ is small or bottleneck metrics are low
2. **Handling**: Scout explicitly generates Macro candidates → PPO handles directly
3. **v4.1 Improvement**: Scout endpoints get bonus for active nodes → explores macro-bridges not in current tour

---

## 4. Scaling Laws

Parameter scaling derived from theory:

```python
mass_threshold = c · log(n)           # Ellipse mass threshold
core_ratio = c · n^(-1/3)             # Exception edge core ratio  
long_threshold = c · √log(n)          # Long edge criterion (v4.1)
max_active_nodes = 2 · n · core_ratio # Focused search node count
```

| n | mass_threshold | core_ratio | max_active_nodes |
|---|----------------|------------|------------------|
| 431 | ~9.1 | ~0.26 | ~226 |
| 1,291 | ~10.7 | ~0.18 | ~474 |
| 5,000 | ~12.8 | ~0.12 | ~1,168 |

---

## 5. Experimental Environment

| Item | Specification |
|------|---------------|
| GPU | NVIDIA A100 |
| Platform | Google Colab |
| Training | On-instance PPO (from scratch) |
| Pretraining | None |
| External Data | None |

### Reproducibility Checklist

- [x] Seed fixed (random, numpy, torch)
- [x] Numba cache (`@njit(cache=True)`)
- [x] Version specified (`__version__ = "4.1.0"`)
- [x] CLI parameter adjustment available
- [x] Log saving (iteration, time, best, mode)

---

## 6. Usage

### Installation

```bash
pip install numpy scipy torch numba matplotlib
```

### CLI Execution

```bash
# Basic run
python tsp_solver.py --input d1291.tsp --time 3000 --verbose

# With seed
python tsp_solver.py --input d1291.tsp --time 3000 --seed 42

# Save result
python tsp_solver.py --input d1291.tsp --time 3000 --output solution.txt
```

### Python API

```python
from tsp_solver import solve, parse_tsplib, SolverConfig

# Load problem
coords, edge_type, meta = parse_tsplib("d1291.tsp")

# Configure
cfg = SolverConfig(
    time_total=3000.0,
    seed=42,
    verbose=True
)

# Solve
result = solve(coords, edge_type=edge_type, cfg=cfg)

print(f"Best tour length: {result['length']}")
```

---

## 7. File Structure

```
exception-edge-tsp/
├── README.md              # Documentation (Korean)
├── README_EN.md           # Documentation (English)
├── tsp_solver.py          # Main solver (single file)
├── requirements.txt       # Dependencies
├── logs/
│   └── d1291_ppo_log.txt  # Experiment log
└── data/
    ├── d1291.tsp          # Test instance
```

---

## 8. License

- **Code**: Apache License 2.0
- **Data**: TSPLIB/public instances follow their respective source policies.

---

## 9. Citation

```bibtex
@software{exception_edge_tsp,
  title={Exception-Edge Theory Guided PPO TSP Solver},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/exception-edge-tsp}
}
```

---

## 10. Key Contributions

1. **Theoretical Framework**: Geometric/topological classification of exception edges
2. **Scaling Laws**: Derivation and verification of mass ~ c·log(n), core ~ c·n^(-1/3)
3. **Zero-shot Learning**: Competitive performance on single instances without pretraining
4. **Interpretability**: Explicit answers to "why is this edge important?"

understanding.

**Q: What makes an edge an "exception"?**

A: An edge is exceptional if:
1. Its detour ratio ρ > 1 + η (no good shortcuts exist), OR
2. It's long AND has low ellipse mass (crosses a structural void)

Both conditions indicate the edge may be necessary in the optimal tour despite not being in the local proximity graph.
