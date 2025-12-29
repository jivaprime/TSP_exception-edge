#!/usr/bin/env python3
"""
Exception-Edge Theory Guided PPO TSP Solver v4.1
Zero-shot on-instance learning, no pretraining required.
"""
from __future__ import annotations
import os, sys, math, time, random, argparse, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set, Any
from collections import defaultdict
from enum import IntEnum
import numpy as np
from scipy.spatial import Delaunay, KDTree
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numba import njit
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

__version__ = "4.1.0"
EDGE_FEAT_DIM, MOVE_FEAT_DIM, SCOUT_FEAT_DIM, META_STATE_DIM = 10, 35, 10, 20
SCOUT_ENDPOINTS: Set[int] = set()

class Mode(IntEnum):
    EXPLORE=0; STANDARD=1; MACRO_FOCUS=2; ESCAPE=3; FINALIZE=4

@dataclass
class TheoryConfig:
    n_nodes: int; c_log: float=1.5; c_core: float=2.0; base_eta: float=0.15; long_len_mult: float=2.2
    def __post_init__(self):
        N = self.n_nodes; self.N = N
        self.mass_threshold = max(6.0, self.c_log * np.log(N))
        self.base_core_ratio = float(np.clip(self.c_core * (N ** (-1/3)), 0.05, 0.40))
        self.eta = float(np.clip(self.base_eta * (1.0 + 0.1 * np.log(N / 100)), 0.10, 0.25))
        self.rho_hi = 1.0 + self.eta; self.rho_lo = 1.0 + 0.5 * self.eta
        self.knn_k = int(np.clip(10 + 5 * np.log(N / 50), 15, 50))
        self.max_active_nodes = int(np.clip(N * self.base_core_ratio * 2, 20, N // 2))
        self.long_threshold = self.long_len_mult * np.sqrt(max(np.log(N), 1.0))
    def is_exception_edge(self, rho, mass, norm_length=0.0):
        if rho > self.rho_hi: return True
        if norm_length >= self.long_threshold and mass < self.mass_threshold: return True
        return False
    def compute_exception_score(self, rho, mass, norm_length):
        rho_score = max(0.0, rho - 1.0)
        gate = 1.0 / (1.0 + np.exp(-(norm_length - self.long_threshold)))
        mass_score = gate * max(0.0, (self.mass_threshold - mass) / (self.mass_threshold + 1e-9))
        return 1.0 * rho_score + 1.2 * mass_score + 0.25 * max(0.0, norm_length - 1.0) / (1.0 + norm_length)

@dataclass
class EdgeInfo:
    u: int; v: int; tour_idx: int; rho: float; mass: int; norm_length: float; exception_score: float; is_exception: bool

@dataclass
class DensityMap:
    local_density: np.ndarray; local_scale: np.ndarray; is_island: np.ndarray; is_sea: np.ndarray
    island_labels: np.ndarray; n_islands: int; density_threshold: float; thickness_score: float=0.0; lambda2: float=0.0

@dataclass
class Move3Opt:
    i: int; j: int; k: int; pattern: int; delta: float; features: np.ndarray; priority_score: float=0.0

@dataclass
class ModeConfig:
    name: str; core_ratio_mult: float; eta_mult: float; macro_budget: int; proposal_M: int
    exception_bias: float; crossover_prob: float; kick_strength: float

@dataclass
class PPOConfig:
    gamma: float=0.99; lam: float=0.95; clip_eps: float=0.2; entropy_coef: float=0.01
    value_coef: float=0.5; train_iters: int=4; max_grad_norm: float=1.0

@dataclass
class SolverConfig:
    time_total: float=300.0; ils_time_ratio: float=0.5; manager_interval: int=50; scout_interval: int=50
    sil_interval: int=200; train_interval: int=100; stagnation_threshold: int=80; elite_pool_size: int=10
    seed: int=42; train: bool=True; verbose: bool=True; ils_ppo_steps: int=100; use_delaunay: bool=True

MODE_PRESETS = [ModeConfig("EXPLORE",0.8,0.9,16,64,0.6,0.3,0.3), ModeConfig("STANDARD",1.0,1.0,32,64,0.7,0.5,0.5),
    ModeConfig("MACRO_FOCUS",0.7,1.3,96,64,0.85,0.7,0.5), ModeConfig("ESCAPE",1.0,1.5,128,64,0.9,0.8,0.9),
    ModeConfig("FINALIZE",1.3,0.85,60,64,0.75,0.6,0.2)]
N_MODES = len(MODE_PRESETS)

@njit(cache=True)
def tour_length_numba(D, tour):
    n, total = len(tour), 0.0
    for i in range(n): total += D[tour[i], tour[(i + 1) % n]]
    return total

@njit(cache=True)
def make_pos_index_numba(tour):
    n = len(tour); pos = np.empty(n, dtype=np.int32)
    for i in range(n): pos[tour[i]] = i
    return pos

@njit(cache=True)
def two_opt_delta_numba(D, tour, i, j):
    n = len(tour); a, b = tour[i], tour[(i + 1) % n]; c, d = tour[j], tour[(j + 1) % n]
    return (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])

@njit(cache=True)
def two_opt_swap_numba(tour, i, j):
    new_tour = tour.copy(); new_tour[i+1:j+1] = tour[i+1:j+1][::-1]; return new_tour

@njit(cache=True)
def three_opt_delta_numba(D, tour, i, j, k, pattern):
    n = len(tour); a, b = tour[i], tour[i+1]; c, d = tour[j], tour[j+1]; e, f = tour[k], tour[(k+1) % n]
    old_cost = D[a,b] + D[c,d] + D[e,f]
    if pattern == 1: new_cost = D[a,c] + D[b,d] + D[e,f]
    elif pattern == 2: new_cost = D[a,b] + D[c,e] + D[d,f]
    elif pattern == 3: new_cost = D[a,c] + D[b,e] + D[d,f]
    elif pattern == 4: new_cost = D[a,d] + D[e,b] + D[c,f]
    elif pattern == 5: new_cost = D[a,d] + D[e,c] + D[b,f]
    elif pattern == 6: new_cost = D[a,e] + D[d,b] + D[c,f]
    elif pattern == 7: new_cost = D[a,e] + D[d,c] + D[b,f]
    else: return 0.0
    return new_cost - old_cost

@njit(cache=True)
def best_3opt_delta_numba(D, tour, i, j, k):
    best_delta, best_pattern = 0.0, 0
    for pat in range(1, 8):
        delta = three_opt_delta_numba(D, tour, i, j, k, pat)
        if delta < best_delta - 1e-9: best_delta, best_pattern = delta, pat
    return best_delta, best_pattern

@njit(cache=True)
def two_opt_numba(D, tour, max_passes=30):
    n, best, improved, passes = len(tour), tour.copy(), True, 0
    while improved and passes < max_passes:
        improved, passes = False, passes + 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: continue
                if two_opt_delta_numba(D, best, i, j) < -1e-9: best = two_opt_swap_numba(best, i, j); improved = True; break
            if improved: break
    return best

@njit(cache=True)
def two_opt_cand_numba(D, tour, cand_flat, cand_ptr, max_passes=30):
    n, best, improved, passes = len(tour), tour.copy(), True, 0
    pos = make_pos_index_numba(best)
    while improved and passes < max_passes:
        improved, passes = False, passes + 1
        for a_idx in range(n - 1):
            a = best[a_idx]
            for ci in range(cand_ptr[a], cand_ptr[a + 1]):
                c = cand_flat[ci]; c_idx = pos[c]
                if c_idx <= a_idx + 1 or c_idx == n - 1: continue
                if two_opt_delta_numba(D, best, a_idx, c_idx) < -1e-9:
                    best = two_opt_swap_numba(best, a_idx, c_idx); pos = make_pos_index_numba(best); improved = True; break
            if improved: break
    return best

@njit(cache=True)
def nn_tour_numba(D, start):
    n = len(D); tour = np.empty(n, dtype=np.int32); visited = np.zeros(n, dtype=np.bool_)
    tour[0] = start; visited[start] = True
    for i in range(1, n):
        cur, best_dist, best_next = tour[i - 1], np.inf, -1
        for j in range(n):
            if not visited[j] and D[cur, j] < best_dist: best_dist, best_next = D[cur, j], j
        tour[i] = best_next; visited[best_next] = True
    return tour

@njit(cache=True)
def double_bridge_kick_numba(tour, cuts):
    n = len(tour); a, b, c, d = cuts[0], cuts[1], cuts[2], cuts[3]
    new_tour = np.empty(n, dtype=np.int32); idx = 0
    for i in range(a): new_tour[idx] = tour[i]; idx += 1
    for i in range(b, c): new_tour[idx] = tour[i]; idx += 1
    for i in range(a, b): new_tour[idx] = tour[i]; idx += 1
    for i in range(c, d): new_tour[idx] = tour[i]; idx += 1
    for i in range(d, n): new_tour[idx] = tour[i]; idx += 1
    return new_tour

def parse_tsplib(path):
    meta, coords, in_node_section, edge_type = {}, [], False, "EUC_2D"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ":" in line and not in_node_section:
                k, v = [s.strip() for s in line.split(":", 1)]; meta[k.upper()] = v
                if k.upper() == "EDGE_WEIGHT_TYPE": edge_type = v.upper()
            elif line.upper().startswith("NODE_COORD_SECTION"): in_node_section = True
            elif line.upper().startswith("EOF"): break
            elif in_node_section:
                parts = line.split()
                if len(parts) >= 3: coords.append((float(parts[1]), float(parts[2])))
    return np.array(coords, dtype=np.float64), edge_type, meta

def build_distance_matrix(coords, edge_type="EUC_2D"):
    if edge_type == "EUC_2D":
        diff = coords[:, None, :] - coords[None, :, :]; return np.sqrt((diff ** 2).sum(axis=2))
    n, D = len(coords), np.zeros((len(coords), len(coords)), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2); D[i, j] = D[j, i] = d
    return D

def build_knn(D, k=20): return np.argsort(D, axis=1)[:, 1:min(k, len(D)-1)+1].astype(np.int32)

def cand_to_numba_format(cand, n):
    flat, ptr = [], [0]
    for i in range(n): flat.extend(sorted(cand[i]) if i < len(cand) else []); ptr.append(len(flat))
    return np.array(flat, dtype=np.int32), np.array(ptr, dtype=np.int32)

def build_candidate_graph(D, knn, coords, scout_bridges=None, use_delaunay=True):
    n = len(D); cand = [set(knn[i].tolist()) for i in range(n)]
    for i in range(n):
        for j in cand[i].copy(): cand[j].add(i)
    if scout_bridges is not None:
        for edge in scout_bridges: u, v = int(edge[0]), int(edge[1]); cand[u].add(v); cand[v].add(u)
    if use_delaunay and n <= 6000:
        try:
            for simplex in Delaunay(coords).simplices:
                a, b, c = simplex; cand[a].add(b); cand[b].add(a); cand[b].add(c); cand[c].add(b); cand[c].add(a); cand[a].add(c)
        except: pass
    return cand

def rho_and_mass_exact(u, v, coords, kdtree, eta):
    pu, pv = coords[u], coords[v]; base_dist = np.linalg.norm(pu - pv)
    if base_dist <= 1e-12: return 1.0, 0
    candidates = kdtree.query_ball_point(0.5 * (pu + pv), r=0.5 * (1.0 + eta) * base_dist)
    candidates = [w for w in candidates if w != u and w != v]
    if not candidates: return 999.0, 0
    W = coords[candidates]; detour = np.linalg.norm(W - pu, axis=1) + np.linalg.norm(W - pv, axis=1)
    return float(np.min(detour / base_dist)), int(np.sum(detour <= (1.0 + eta) * base_dist))

def compute_all_edge_info(tour, coords, D, kdtree, local_scale, theory_cfg):
    n, edge_infos = len(tour), []
    for i in range(n):
        u, v = int(tour[i]), int(tour[(i + 1) % n])
        rho, mass = rho_and_mass_exact(u, v, coords, kdtree, theory_cfg.eta)
        norm_length = D[u, v] / (min(local_scale[u], local_scale[v]) + 1e-9)
        edge_infos.append(EdgeInfo(u, v, i, rho, mass, norm_length, theory_cfg.compute_exception_score(rho, mass, norm_length), theory_cfg.is_exception_edge(rho, mass, norm_length)))
    return edge_infos

def identify_active_nodes(edge_infos, theory_cfg):
    global SCOUT_ENDPOINTS
    score = defaultdict(float)
    for ei in edge_infos:
        if ei.is_exception: score[ei.u] += ei.exception_score; score[ei.v] += ei.exception_score
    for u in SCOUT_ENDPOINTS: score[u] += 2.5
    if not score:
        active = set()
        for ei in edge_infos:
            if ei.is_exception: active.add(ei.u); active.add(ei.v)
        return active
    return set(u for u, _ in sorted(score.items(), key=lambda x: -x[1])[:theory_cfg.max_active_nodes])

def compute_density_map(coords, D, knn, k_density=8, quantile=0.3):
    n = len(coords); sorted_D = np.sort(D, axis=1); local_scale = np.maximum(sorted_D[:, min(k_density, n-1)], 1e-9)
    local_density = 1.0 / (local_scale ** 2 + 1e-12); density_threshold = np.quantile(local_density, quantile)
    is_sea = local_density < density_threshold; is_island = ~is_sea; island_labels = np.full(n, -1, dtype=np.int32)
    island_nodes = np.where(is_island)[0]; label = 0
    if len(island_nodes) > 3:
        try:
            adj = defaultdict(set)
            for simplex in Delaunay(coords[island_nodes]).simplices:
                for ii in range(3):
                    for jj in range(ii + 1, 3):
                        u, v = island_nodes[simplex[ii]], island_nodes[simplex[jj]]
                        if D[u, v] <= 2.0 * max(local_scale[u], local_scale[v]): adj[u].add(v); adj[v].add(u)
            visited = set()
            for start in island_nodes:
                if start in visited: continue
                queue = [start]; visited.add(start)
                while queue:
                    node = queue.pop(0); island_labels[node] = label
                    for nb in adj[node]:
                        if nb not in visited: visited.add(nb); queue.append(nb)
                label += 1
        except: island_labels[island_nodes] = 0; label = 1
    elif len(island_nodes) > 0: island_labels[island_nodes] = 0; label = 1
    k = knn.shape[1]; row_idx = np.repeat(np.arange(n), k); col_idx = knn.flatten()
    adj_sp = sp.csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(n, n))
    adj_sp = adj_sp + adj_sp.T; adj_sp.data = np.clip(adj_sp.data, 0, 1)
    deg = np.array(adj_sp.sum(axis=1)).reshape(-1); deg[deg == 0] = 1.0
    L = sp.eye(n) - sp.diags(1.0 / np.sqrt(deg)) @ adj_sp @ sp.diags(1.0 / np.sqrt(deg))
    try: lambda2 = float(np.sort(eigsh(L, k=2, which="SM", return_eigenvectors=False))[1])
    except: lambda2 = 0.1
    return DensityMap(local_density, local_scale, is_island, is_sea, island_labels, label, density_threshold, 1.0/(lambda2+1e-6), lambda2)

def tour_to_edges(tour): return set((min(int(tour[i]), int(tour[(i+1)%len(tour)])), max(int(tour[i]), int(tour[(i+1)%len(tour)]))) for i in range(len(tour)))
def edges_to_adj(edges, n):
    adj = defaultdict(list)
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    return adj

def eax_crossover(tour_a, tour_b, D, rng):
    n = len(tour_a); edges_a, edges_b = tour_to_edges(tour_a), tour_to_edges(tour_b); common = edges_a & edges_b
    if len(common) > n * 0.95: return tour_a.copy()
    adj = edges_to_adj(common, n); degree = {i: len(adj.get(i, [])) for i in range(n)}
    repair = list((edges_a - common) | (edges_b - common)); rng.shuffle(repair)
    for u, v in repair:
        if degree[u] < 2 and degree[v] < 2 and v not in adj.get(u, []): adj[u].append(v); adj[v].append(u); degree[u] += 1; degree[v] += 1
    endpoints = [i for i in range(n) if degree[i] < 2]
    while len(endpoints) >= 2:
        u = endpoints.pop(0)
        if degree[u] >= 2: continue
        best_v, best_d = None, float('inf')
        for v in endpoints:
            if v != u and degree[v] < 2 and v not in adj.get(u, []) and D[u,v] < best_d: best_v, best_d = v, D[u,v]
        if best_v: adj[u].append(best_v); adj[best_v].append(u); degree[u] += 1; degree[best_v] += 1; endpoints = [x for x in endpoints if degree[x] < 2]
    result_edges = set((min(u, v), max(u, v)) for u in adj for v in adj[u])
    adj2 = edges_to_adj(result_edges, n)
    for node in range(n):
        if len(adj2.get(node, [])) != 2: return tour_a.copy()
    tour, visited, current = [0], {0}, 0
    while len(tour) < n:
        next_node = None
        for nb in adj2[current]:
            if nb not in visited: next_node = nb; break
        if next_node is None: return tour_a.copy()
        tour.append(next_node); visited.add(next_node); current = next_node
    return np.array(tour, dtype=np.int32)

class ElitePool:
    def __init__(self, max_size=10, div_th=0.9): self.max_size, self.div_th, self.pool = max_size, div_th, []
    def add(self, tour, length):
        edges = tour_to_edges(tour)
        for idx, (el, _, ee) in enumerate(self.pool):
            if len(edges & ee) / len(edges) > self.div_th:
                if length < el: self.pool[idx] = (length, tour.copy(), edges); self.pool.sort(key=lambda x: x[0])
                return
        self.pool.append((length, tour.copy(), edges)); self.pool.sort(key=lambda x: x[0])
        if len(self.pool) > self.max_size: self.pool.pop()
    def get_random_partner(self, exclude_tour, rng):
        if len(self.pool) < 2: return None
        ee = tour_to_edges(exclude_tour) if exclude_tour is not None else set()
        cands = [t for _, t, e in self.pool if len(e & ee) / max(len(e), 1) < 0.98]
        return cands[rng.randint(0, len(cands)-1)] if cands else self.pool[rng.randint(0, len(self.pool)-1)][1]
    def __len__(self): return len(self.pool)
    def get_all_edges(self): return set().union(*(e for _, _, e in self.pool)) if self.pool else set()

def generate_macro_candidates(coords, D, density_map, knn, kdtree, theory_cfg, max_cands=300):
    global SCOUT_ENDPOINTS
    n, candidates = len(coords), []
    islands = defaultdict(list)
    for i in range(n):
        if density_map.island_labels[i] >= 0: islands[density_map.island_labels[i]].append(i)
    labels = list(islands.keys())
    for i_l, li in enumerate(labels):
        for lj in labels[i_l+1:]:
            ni, nj = islands[li], islands[lj]
            pairs = [(D[random.choice(ni), random.choice(nj)], random.choice(ni), random.choice(nj)) for _ in range(min(200, len(ni)*len(nj)))]
            pairs.sort(); candidates.extend([(u, v) for _, u, v in pairs[:20]])
    sea = np.where(density_map.is_sea)[0]
    for _ in range(min(100, max_cands//3)):
        if len(sea) > 0:
            u = random.choice(sea); knn_set = set(knn[u].tolist())
            outside = [j for j in range(n) if j != u and j not in knn_set]
            if outside: candidates.append((u, random.choice(outside)))
    candidates = list(set((min(u,v), max(u,v)) for u, v in candidates))
    scored = []
    for u, v in candidates[:max_cands * 2]:
        rho, mass = rho_and_mass_exact(u, v, coords, kdtree, theory_cfg.eta)
        norm_length = D[u, v] / (min(density_map.local_scale[u], density_map.local_scale[v]) + 1e-9)
        score = theory_cfg.compute_exception_score(rho, mass, norm_length) + (2.0 if theory_cfg.is_exception_edge(rho, mass, norm_length) else 0)
        scored.append((u, v, score, rho, mass))
    scored.sort(key=lambda x: -x[2]); scored = scored[:max_cands]
    if not scored: scored = [(random.randint(0,n-1), random.randint(0,n-1), 0.0, 1.0, 0) for _ in range(50)]
    endpoints = []
    for u, v, _, _, _ in scored: endpoints.extend([u, v]); 
        if len(endpoints) >= 80: break
    SCOUT_ENDPOINTS = set(endpoints)
    edges = np.array([(u, v) for u, v, _, _, _ in scored], dtype=np.int32)
    features = np.zeros((len(edges), SCOUT_FEAT_DIM), dtype=np.float32)
    for i, (u, v, _, rho, mass) in enumerate(scored):
        norm_length = D[u, v] / (min(density_map.local_scale[u], density_map.local_scale[v]) + 1e-9)
        features[i] = [np.log1p(D[u,v]), norm_length, min(rho, 5.0), mass/(theory_cfg.mass_threshold+1e-9),
                       float(mass < theory_cfg.mass_threshold), float(rho > theory_cfg.rho_hi),
                       float(density_map.is_sea[u] or density_map.is_sea[v]),
                       float(density_map.island_labels[u] != density_map.island_labels[v]),
                       density_map.local_density[u], density_map.local_density[v]]
    return edges, features

class ScoutNetwork(nn.Module):
    def __init__(self, feat_dim=SCOUT_FEAT_DIM, hidden=64):
        super().__init__()
        self.scorer = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x): return self.scorer(x).squeeze(-1)
    def select_top_k(self, x, k=50): return torch.topk(self.forward(x), min(k, len(x)))[1]

def sil_update_scout(scout, opt, edges, feats, best_tour, elite_pool, device, n_steps=5):
    best_edges, elite_edges = tour_to_edges(best_tour), elite_pool.get_all_edges()
    labels = np.array([1.0 if (min(int(e[0]),int(e[1])),max(int(e[0]),int(e[1]))) in best_edges else 0.5 if (min(int(e[0]),int(e[1])),max(int(e[0]),int(e[1]))) in elite_edges else 0.0 for e in edges], dtype=np.float32)
    if labels.sum() < 1: return
    feats_t, labels_t = torch.tensor(feats, dtype=torch.float32, device=device), torch.tensor(labels, dtype=torch.float32, device=device)
    for _ in range(n_steps):
        loss = F.binary_cross_entropy_with_logits(scout(feats_t), labels_t); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(scout.parameters(), 1.0); opt.step()

def compute_move_features(tour, i, j, k, edge_infos, density_map, theory_cfg, delta, n):
    feats = []
    for idx in [i, j, k % n]:
        ei = edge_infos[idx]
        feats.extend([np.log1p(ei.norm_length), min(ei.rho, 5.0), ei.mass/(theory_cfg.mass_threshold+1e-9), ei.exception_score, float(ei.is_exception),
                      float(density_map.is_sea[ei.u] or density_map.is_sea[ei.v]), float(density_map.island_labels[ei.u] != density_map.island_labels[ei.v]),
                      density_map.local_density[ei.u], density_map.local_density[ei.v], ei.norm_length])
    feats.extend([float(i)/n, float(j)/n, float(k)/n, delta, delta/1000.0])
    return np.array(feats, dtype=np.float32)

def propose_moves_exception_biased(tour, D, edge_infos, active_nodes, knn, cand, density_map, theory_cfg, M, rng):
    n, pos = len(tour), make_pos_index_numba(tour)
    active_indices = [pos[node] for node in active_nodes if node < n] or list(range(n))
    edge_weights = [ei.exception_score + 0.1 for ei in edge_infos]; total = sum(edge_weights); edge_weights = [w/total for w in edge_weights]
    moves, seen, attempts = [], set(), 0
    while len(moves) < M and attempts < M * 50:
        attempts += 1
        i = edge_infos[rng.choices(range(n), weights=edge_weights, k=1)[0]].tour_idx if rng.random() < 0.8 else rng.choice(active_indices)
        if i >= n - 5: continue
        a = int(tour[i]); nb_j = list(cand[a]) if a < len(cand) else []
        active_nb = [v for v in nb_j if v in active_nodes]
        j_node = rng.choice(active_nb) if active_nb and rng.random() < 0.7 else (rng.choice(nb_j) if nb_j else None)
        if j_node is None: continue
        j = int(pos[j_node])
        if not (i + 2 <= j <= n - 4): continue
        b = int(tour[j]); nb_k = list(cand[b]) if b < len(cand) else []
        active_nb_k = [v for v in nb_k if v in active_nodes]
        k_node = rng.choice(active_nb_k) if active_nb_k and rng.random() < 0.7 else (rng.choice(nb_k) if nb_k else None)
        if k_node is None: continue
        k = int(pos[k_node])
        if not (j + 2 <= k <= n - 2) or (i, j, k) in seen: continue
        seen.add((i, j, k))
        delta, pattern = best_3opt_delta_numba(D, tour, i, j, k)
        if pattern == 0: continue
        moves.append(Move3Opt(i, j, k, pattern, delta, compute_move_features(tour, i, j, k, edge_infos, density_map, theory_cfg, delta, n),
                              edge_infos[i].exception_score + edge_infos[j].exception_score + edge_infos[k%n].exception_score))
    if not moves:
        i, j, k = 0, n//3, 2*n//3; delta, pattern = best_3opt_delta_numba(D, tour, i, j, k)
        moves.append(Move3Opt(i, j, k, max(pattern, 1), delta, np.zeros(MOVE_FEAT_DIM, dtype=np.float32), 0.0))
    while len(moves) < M: moves.append(moves[rng.randint(0, len(moves)-1)])
    return moves[:M]

def apply_3opt(tour, i, j, k, pattern):
    S1, S2, S3, S4 = tour[:i+1], tour[i+1:j+1], tour[j+1:k+1], tour[k+1:]
    patterns = [tour, np.concatenate([S1, S2[::-1], S3, S4]), np.concatenate([S1, S2, S3[::-1], S4]), np.concatenate([S1, S2[::-1], S3[::-1], S4]),
                np.concatenate([S1, S3, S2, S4]), np.concatenate([S1, S3[::-1], S2, S4]), np.concatenate([S1, S3, S2[::-1], S4]), np.concatenate([S1, S3[::-1], S2[::-1], S4])]
    return patterns[pattern].astype(np.int32)

class WorkerPolicy(nn.Module):
    def __init__(self, raw_dim=MOVE_FEAT_DIM, emb_dim=32, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, emb_dim), nn.Tanh())
        self.pi = nn.Sequential(nn.Linear(emb_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.v = nn.Sequential(nn.Linear(emb_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        B, M, R = x.shape; enc = self.encoder(x.view(B*M, R)).view(B, M, -1)
        return self.pi(enc).squeeze(-1).squeeze(0), self.v(enc.mean(dim=1)).squeeze(0)

class ManagerPolicy(nn.Module):
    def __init__(self, state_dim=META_STATE_DIM, n_actions=N_MODES, hidden=128):
        super().__init__()
        self.pi = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))
        self.v = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        return self.pi(x).squeeze(0), self.v(x).squeeze(0)

def compute_meta_state(best_len, cur_len, init_len, it, stagnation, time_elapsed, time_total, mode_cfg, edge_infos, density_map, theory_cfg, elite_pool, n):
    exc_count = sum(1 for ei in edge_infos if ei.is_exception); exc_ratio = exc_count / max(len(edge_infos), 1)
    avg_exc = np.mean([ei.exception_score for ei in edge_infos]) if edge_infos else 0
    mode_idx = MODE_PRESETS.index(mode_cfg) if mode_cfg in MODE_PRESETS else 0
    return np.array([math.log(best_len+1), (cur_len-best_len)/(best_len+1e-9), (init_len-best_len)/(init_len+1e-9), it/1000.0, min(stagnation/500.0, 1.0),
                     time_elapsed/(time_total+1e-9), exc_ratio, avg_exc/5.0, density_map.n_islands/10.0, np.sum(density_map.is_sea)/n, mode_idx/N_MODES,
                     mode_cfg.core_ratio_mult, mode_cfg.eta_mult, mode_cfg.exception_bias, len(elite_pool)/10.0, density_map.thickness_score/100.0,
                     density_map.lambda2, np.log(n)/10.0, theory_cfg.mass_threshold/20.0, theory_cfg.base_core_ratio], dtype=np.float32)

class RolloutBuffer:
    def __init__(self): self.clear()
    def add(self, obs, act, logp, val, rew, done): self.obs.append(obs); self.act.append(act); self.logp.append(logp); self.val.append(val); self.rew.append(rew); self.done.append(done)
    def __len__(self): return len(self.rew)
    def clear(self): self.obs, self.act, self.logp, self.val, self.rew, self.done = [], [], [], [], [], []
    def compute_gae(self, gamma, lam):
        T = len(self.rew)
        if T == 0: return np.array([]), np.array([])
        val, rew, done = np.array(self.val), np.array(self.rew), np.array(self.done); adv, last_gae = np.zeros(T), 0.0
        for t in reversed(range(T)):
            next_val = 0.0 if t == T-1 else val[t+1] * (1 - done[t+1])
            delta = rew[t] + gamma * next_val - val[t]; last_gae = delta + gamma * lam * (1 - done[t]) * last_gae; adv[t] = last_gae
        ret = adv + val
        if adv.std() > 1e-8: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv.astype(np.float32), ret.astype(np.float32)

def ppo_update(policy, opt, buffer, cfg, device, is_manager=False):
    if len(buffer) < 4: return
    adv, ret = buffer.compute_gae(cfg.gamma, cfg.lam)
    if len(adv) == 0: return
    obs, act, logp_old = np.array(buffer.obs), np.array(buffer.act), np.array(buffer.logp)
    for _ in range(cfg.train_iters):
        obs_t, act_t = torch.tensor(obs, dtype=torch.float32, device=device), torch.tensor(act, dtype=torch.int64, device=device)
        logp_old_t, adv_t, ret_t = torch.tensor(logp_old, dtype=torch.float32, device=device), torch.tensor(adv, dtype=torch.float32, device=device), torch.tensor(ret, dtype=torch.float32, device=device)
        if is_manager: logits, val = policy(obs_t); val = val.view(-1)
        else: logits_list, val_list = [], []; [logits_list.append(policy(obs_t[i])[0]) or val_list.append(policy(obs_t[i])[1]) for i in range(len(obs))]; logits, val = torch.stack(logits_list), torch.stack(val_list).view(-1)
        dist = torch.distributions.Categorical(logits=logits); logp = dist.log_prob(act_t); ratio = torch.exp(logp - logp_old_t)
        loss = -torch.min(ratio * adv_t, torch.clamp(ratio, 1-cfg.clip_eps, 1+cfg.clip_eps) * adv_t).mean() + cfg.value_coef * F.mse_loss(val, ret_t) - cfg.entropy_coef * dist.entropy().mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm); opt.step()

def generate_kick_cuts(n, rng): return np.array(sorted(rng.sample(range(max(1, n//10), n - max(1, n//10)), 4)), dtype=np.int32)

def ppo_3opt_phase(tour, D, coords, kdtree, density_map, theory_cfg, worker, cand, cand_flat, cand_ptr, n_steps, rng, device):
    n, knn = len(tour), build_knn(D, k=theory_cfg.knn_k)
    for _ in range(n_steps):
        edge_infos = compute_all_edge_info(tour, coords, D, kdtree, density_map.local_scale, theory_cfg)
        active_nodes = identify_active_nodes(edge_infos, theory_cfg)
        moves = propose_moves_exception_biased(tour, D, edge_infos, active_nodes, knn, cand, density_map, theory_cfg, 32, rng)
        if not moves: continue
        with torch.no_grad(): action = int(torch.distributions.Categorical(logits=worker(torch.tensor(np.stack([m.features for m in moves]), dtype=torch.float32, device=device))[0]).sample().item())
        if moves[action].delta < -1e-9: tour = two_opt_cand_numba(D, apply_3opt(tour, moves[action].i, moves[action].j, moves[action].k, moves[action].pattern), cand_flat, cand_ptr, max_passes=5)
    return tour

def solve(coords, edge_type="EUC_2D", cfg=None, verbose=None):
    global SCOUT_ENDPOINTS; SCOUT_ENDPOINTS = set()
    if cfg is None: cfg = SolverConfig()
    if verbose is not None: cfg.verbose = verbose
    rng = random.Random(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    n, device, ils_time = len(coords), "cuda" if torch.cuda.is_available() else "cpu", cfg.time_total * cfg.ils_time_ratio
    if cfg.verbose: print(f"Exception-Edge PPO TSP Solver v{__version__} | n={n}, time={cfg.time_total}s")
    theory_cfg = TheoryConfig(n_nodes=n)
    D, knn, kdtree = build_distance_matrix(coords, edge_type), build_knn(build_distance_matrix(coords, edge_type), k=theory_cfg.knn_k), KDTree(coords)
    density_map = compute_density_map(coords, D, knn)
    tour = two_opt_numba(D, nn_tour_numba(D, rng.randint(0, n-1)), max_passes=50)
    init_len, after_2opt = tour_length_numba(D, nn_tour_numba(D, 0)), tour_length_numba(D, tour)
    best_tour, best_len, cur_len = tour.copy(), after_2opt, after_2opt
    elite_pool = ElitePool(max_size=cfg.elite_pool_size); elite_pool.add(best_tour, best_len)
    worker, manager, scout = WorkerPolicy().to(device), ManagerPolicy().to(device), ScoutNetwork().to(device)
    worker_opt, manager_opt, scout_opt = optim.Adam(worker.parameters(), lr=3e-4), optim.Adam(manager.parameters(), lr=1e-4), optim.Adam(scout.parameters(), lr=2e-4)
    ppo_cfg, worker_buf, manager_buf = PPOConfig(), RolloutBuffer(), RolloutBuffer()
    global_start, main_end, mode_cfg = time.time(), time.time() + cfg.time_total - ils_time, MODE_PRESETS[Mode.STANDARD]
    loose_edges, loose_feats = generate_macro_candidates(coords, D, density_map, knn, kdtree, theory_cfg, max_cands=300)
    cand = build_candidate_graph(D, knn, coords, loose_edges[:mode_cfg.macro_budget] if len(loose_edges) > 0 else None, cfg.use_delaunay)
    cand_flat, cand_ptr = cand_to_numba_format(cand, n)
    edge_infos, active_nodes = compute_all_edge_info(tour, coords, D, kdtree, density_map.local_scale, theory_cfg), identify_active_nodes(compute_all_edge_info(tour, coords, D, kdtree, density_map.local_scale, theory_cfg), theory_cfg)
    it, stagnation, last_manager_best, eax_count, kick_count, logs = 0, 0, best_len, 0, 0, []
    if cfg.verbose: print(f"[Init] NN={init_len:.2f}, 2-opt={after_2opt:.2f}")
    while time.time() < main_end:
        elapsed = time.time() - global_start
        if it % 20 == 0: edge_infos = compute_all_edge_info(tour, coords, D, kdtree, density_map.local_scale, theory_cfg); active_nodes = identify_active_nodes(edge_infos, theory_cfg)
        if it % cfg.manager_interval == 0:
            meta_state = compute_meta_state(best_len, cur_len, init_len, it, stagnation, elapsed, cfg.time_total, mode_cfg, edge_infos, density_map, theory_cfg, elite_pool, n)
            with torch.no_grad(): logits, value = manager(torch.tensor(meta_state, dtype=torch.float32, device=device)); dist = torch.distributions.Categorical(logits=logits); action = int(dist.sample().item()); logp, val = float(dist.log_prob(torch.tensor(action, device=device)).item()), float(value.item())
            old_mode = mode_cfg.name; mode_cfg = MODE_PRESETS[action]
            if len(manager_buf) > 0: manager_buf.rew[-1] = (last_manager_best - best_len) / (last_manager_best + 1e-9) * 100
            manager_buf.add(meta_state, action, logp, val, 0.0, False); last_manager_best = best_len
            if cfg.verbose and old_mode != mode_cfg.name: print(f"  [{elapsed:.1f}s] Mode: {old_mode} → {mode_cfg.name}")
        if it % cfg.scout_interval == 0:
            if it % (cfg.scout_interval * 3) == 0: loose_edges, loose_feats = generate_macro_candidates(coords, D, density_map, knn, kdtree, theory_cfg, max_cands=300)
            if len(loose_edges) > 0:
                with torch.no_grad(): indices = scout.select_top_k(torch.tensor(loose_feats, dtype=torch.float32, device=device), k=mode_cfg.macro_budget)
                cand = build_candidate_graph(D, knn, coords, loose_edges[indices.cpu().numpy()], cfg.use_delaunay); cand_flat, cand_ptr = cand_to_numba_format(cand, n)
        if cfg.train and it % cfg.sil_interval == 0 and it > 0 and len(loose_edges) > 0: sil_update_scout(scout, scout_opt, loose_edges, loose_feats, best_tour, elite_pool, device)
        moves = propose_moves_exception_biased(tour, D, edge_infos, active_nodes, knn, cand, density_map, theory_cfg, mode_cfg.proposal_M, rng)
        if not moves: stagnation += 1; it += 1; continue
        obs = np.stack([m.features for m in moves])
        with torch.no_grad(): logits, value = worker(torch.tensor(obs, dtype=torch.float32, device=device)); dist = torch.distributions.Categorical(logits=logits); action = int(dist.sample().item()); logp, val = float(dist.log_prob(torch.tensor(action, device=device)).item()), float(value.item())
        chosen = moves[action]; new_tour = two_opt_cand_numba(D, apply_3opt(tour, chosen.i, chosen.j, chosen.k, chosen.pattern), cand_flat, cand_ptr, max_passes=10); new_len = tour_length_numba(D, new_tour)
        reward = (cur_len - new_len) / (cur_len + 1e-9) * 100 - 0.01; tour, cur_len = new_tour, new_len
        if cur_len < best_len - 1e-9: best_tour, best_len, stagnation = tour.copy(), cur_len, 0; elite_pool.add(best_tour, best_len);
            if cfg.verbose: print(f"  [{elapsed:.1f}s] it={it} best={best_len:.2f}")
        else: stagnation += 1
        worker_buf.add(obs, action, logp, val, reward, False)
        if stagnation > cfg.stagnation_threshold:
            if rng.random() < mode_cfg.crossover_prob and len(elite_pool) >= 2:
                partner = elite_pool.get_random_partner(tour, rng)
                if partner is not None: tour = eax_crossover(best_tour, partner, D, rng); eax_count += 1
                else: tour = double_bridge_kick_numba(best_tour, generate_kick_cuts(n, rng)); kick_count += 1
            else: tour = double_bridge_kick_numba(best_tour, generate_kick_cuts(n, rng)); kick_count += 1
            tour = two_opt_numba(D, tour, max_passes=20); cur_len = tour_length_numba(D, tour); stagnation = 0
        if cfg.train and it % cfg.train_interval == 0 and it > 0:
            if len(worker_buf) >= 50: ppo_update(worker, worker_opt, worker_buf, ppo_cfg, device); worker_buf.clear()
            if len(manager_buf) >= 6: ppo_update(manager, manager_opt, manager_buf, ppo_cfg, device, is_manager=True); manager_buf.clear()
        if it % 30 == 0: logs.append({"time": elapsed, "it": it, "best": best_len, "mode": mode_cfg.name})
        it += 1
    main_loop_best = best_len
    if cfg.verbose: print(f"\n[ILS] ({ils_time}s)")
    tour, ils_start, ils_it = best_tour.copy(), time.time(), 0
    ils_cand = build_candidate_graph(D, knn, coords, loose_edges[:60] if len(loose_edges) > 0 else None, cfg.use_delaunay); ils_cand_flat, ils_cand_ptr = cand_to_numba_format(ils_cand, n)
    while time.time() < ils_start + ils_time:
        partner = elite_pool.get_random_partner(tour, rng) if rng.random() < 0.7 and len(elite_pool) >= 2 else None
        trial = eax_crossover(tour, partner, D, rng) if partner else double_bridge_kick_numba(tour, generate_kick_cuts(n, rng))
        trial = ppo_3opt_phase(trial, D, coords, kdtree, density_map, theory_cfg, worker, ils_cand, ils_cand_flat, ils_cand_ptr, cfg.ils_ppo_steps, rng, device)
        trial = two_opt_numba(D, two_opt_cand_numba(D, trial, ils_cand_flat, ils_cand_ptr, max_passes=30), max_passes=20); trial_len = tour_length_numba(D, trial)
        elite_pool.add(trial, trial_len)
        if trial_len < best_len - 1e-9: tour, best_tour, best_len = trial, trial.copy(), trial_len;
            if cfg.verbose: print(f"  [ILS {time.time()-ils_start:.1f}s] best={best_len:.2f}")
        ils_it += 1
    total_time = time.time() - global_start
    if cfg.verbose: print(f"\n[Final] {best_len:.2f} (improvement={100*(init_len-best_len)/init_len:.2f}%, time={total_time:.1f}s)")
    return {"tour": best_tour, "length": best_len, "init_length": init_len, "after_2opt": after_2opt, "main_loop_best": main_loop_best, "iterations": it, "ils_iterations": ils_it, "eax_count": eax_count, "kick_count": kick_count, "time": total_time, "logs": logs}

def main():
    parser = argparse.ArgumentParser(description="Exception-Edge PPO TSP Solver")
    parser.add_argument("--input", "-i", required=True); parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--time", "-t", type=float, default=300); parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true"); parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()
    if not os.path.exists(args.input): print(f"Error: {args.input} not found"); sys.exit(1)
    coords, edge_type, meta = parse_tsplib(args.input); print(f"Problem: {meta.get('NAME', args.input)}, n={len(coords)}")
    result = solve(coords, edge_type=edge_type, cfg=SolverConfig(time_total=args.time, seed=args.seed, verbose=args.verbose))
    print(f"Best: {result['length']:.2f}")
    if args.output:
        with open(args.output, "w") as f: f.write(f"# Length: {result['length']}\n"); [f.write(f"{node + 1}\n") for node in result["tour"]]

if __name__ == "__main__": main()
