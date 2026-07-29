from __future__ import annotations

from dataclasses import dataclass
from math import inf
from operator import index
from typing import Iterable, Sequence

import numpy as np

from .geometry import Edge


_BINARY64_EXACT_INTEGER_LIMIT = float(2**53)


def _guard_exact_integer_dual_arithmetic(
    matrix: np.ndarray,
    potentials: np.ndarray,
) -> None:
    """Reject integer-mode dual states that could exceed exact binary64."""

    if not np.array_equal(matrix, np.rint(matrix)):
        raise AssertionError("integer arithmetic guard received raw distances")
    if not np.all(np.isfinite(potentials)) or not np.array_equal(
        potentials,
        np.rint(potentials),
    ):
        raise OverflowError("integer potentials lost exact integrality")
    n = len(matrix)
    max_distance = float(np.max(np.abs(matrix)))
    max_potential = float(np.max(np.abs(potentials)))
    # A root 1-tree has n selected edges.  This conservative bound covers
    # every modified edge, its sequential sum, 2*sum(pi), and the final
    # subtraction without relying on cancellation.
    max_modified_edge = max_distance + 2.0 * max_potential
    worst_intermediate = (
        n * max_modified_edge + 2.0 * n * max_potential
    )
    if (
        max_modified_edge >= _BINARY64_EXACT_INTEGER_LIMIT
        or worst_intermediate >= _BINARY64_EXACT_INTEGER_LIMIT
    ):
        raise OverflowError(
            "integer Held--Karp dual arithmetic may exceed the exact "
            "binary64 range"
        )


@dataclass(frozen=True)
class EndpointHamiltonianPath:
    """Exact fixed-endpoint Hamiltonian path in an allowed-edge graph.

    ``path`` is oriented from the smaller endpoint to the larger endpoint.
    An infeasible endpoint pair has value ``inf`` and an empty path.
    """

    value: float
    path: tuple[int, ...]

    @property
    def feasible(self) -> bool:
        return bool(self.path)


@dataclass(frozen=True)
class PairClosureThreshold:
    """Exact-small q=1 closure value and a theorem-safe gain interval.

    The exact gain is ``Z0 - H_e - c(e)``.  The lower and upper gains use
    independently checkable feasible witnesses and 1-tree lower bounds:

    ``cycle_lower - path_upper - c(e) <= exact_gain``
    ``exact_gain <= cycle_upper - path_lower - c(e)``.

    The inequalities are exact-arithmetic statements.  A floating-point study
    must audit them with an explicit numerical guard; integer objectives whose
    sums stay below ``2**53`` are exactly represented by binary64.
    """

    edge: Edge
    edge_cost: float
    path_exact: float
    path_witness: tuple[int, ...]
    closure_exact: float
    cycle_lower: float
    cycle_one_tree_lower: float
    cycle_held_karp_lower: float
    cycle_upper: float
    path_lower: float
    path_one_tree_lower: float
    path_held_karp_lower: float
    path_upper: float
    release_exact: float
    gain_exact: float
    gain_lower: float
    gain_upper: float
    kappa_exact: float
    kappa_lower: float
    kappa_upper: float


@dataclass(frozen=True)
class PairClosureThresholdAnalysis:
    """All nonbaseline q=1 closures for one baseline graph."""

    cycle_exact: float
    cycle_witness: tuple[int, ...]
    cycle_lower: float
    cycle_one_tree_lower: float
    cycle_held_karp_lower: float
    pairs: tuple[PairClosureThreshold, ...]


def _validate_distance_matrix(dist: np.ndarray) -> np.ndarray:
    matrix = np.asarray(dist, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dist must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError("a Hamiltonian path requires at least two vertices")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("dist must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError("dist must be nonnegative")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("dist must be exactly symmetric")
    return matrix


def _normalize_allowed_edges(
    n: int,
    allowed_edges: Iterable[Edge],
) -> set[Edge]:
    try:
        iterator = iter(allowed_edges)
    except TypeError as exc:
        raise ValueError(
            "allowed_edges must be an iterable of edge pairs"
        ) from exc

    normalized: set[Edge] = set()
    for candidate in iterator:
        try:
            raw_u, raw_v = candidate
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each allowed edge must contain exactly two endpoints"
            ) from exc
        try:
            u, v = index(raw_u), index(raw_v)
        except TypeError as exc:
            raise ValueError("allowed edge endpoints must be integers") from exc
        if u == v:
            raise ValueError("allowed edges cannot contain self-loops")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(
                "allowed edge endpoint is outside the distance matrix"
            )
        normalized.add((u, v) if u < v else (v, u))
    return normalized


def _normalize_endpoint_pair(n: int, endpoints: Edge) -> Edge:
    try:
        raw_s, raw_t = endpoints
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "endpoints must contain exactly two vertices"
        ) from exc
    try:
        s, t = index(raw_s), index(raw_t)
    except TypeError as exc:
        raise ValueError("endpoint vertices must be integers") from exc
    if s == t or not (0 <= s < n and 0 <= t < n):
        raise ValueError(
            "endpoints must be two distinct matrix vertices"
        )
    return (s, t) if s < t else (t, s)


def held_karp_all_endpoint_paths(
    dist: np.ndarray,
    allowed_edges: Iterable[Edge],
) -> dict[Edge, EndpointHamiltonianPath]:
    """Solve every fixed-endpoint Hamiltonian-path problem exactly.

    For each unordered endpoint pair ``(s, t)``, ``s < t``, this returns the
    minimum-cost spanning Hamiltonian path from ``s`` to ``t`` using only
    ``allowed_edges``.  Every pair is present in the result; infeasible pairs
    map to ``EndpointHamiltonianPath(inf, ())``.

    One sparse Held--Karp dynamic program is run for each smaller endpoint
    ``s``.  Equal-cost states retain the lexicographically smallest oriented
    path, so witnesses are deterministic and independent of the input edge
    order.  With a complete allowed graph, scalar-state complexity is
    ``O(n^3 2^n)`` time across all starts and ``O(n 2^n)`` space for one
    start.  Storing path tuples for deterministic witnesses adds a
    conservative factor ``O(n)`` to Python-level time and space.
    """

    matrix = _validate_distance_matrix(dist)
    n = int(matrix.shape[0])
    allowed = _normalize_allowed_edges(n, allowed_edges)

    adjacency: tuple[tuple[int, ...], ...] = tuple(
        tuple(
            sorted(
                v if u == vertex else u
                for u, v in allowed
                if u == vertex or v == vertex
            )
        )
        for vertex in range(n)
    )
    result: dict[Edge, EndpointHamiltonianPath] = {
        (s, t): EndpointHamiltonianPath(inf, ())
        for s in range(n)
        for t in range(s + 1, n)
    }
    full_mask = (1 << n) - 1

    # Only starts that can be the smaller member of an unordered pair are
    # needed.  A layer maps the last vertex to (cost, oriented path).
    for start in range(n - 1):
        start_mask = 1 << start
        layers: list[
            dict[int, tuple[float, tuple[int, ...]]]
        ] = [{} for _ in range(1 << n)]
        layers[start_mask][start] = (0.0, (start,))

        for mask in range(1 << n):
            if not (mask & start_mask):
                continue
            for last, (current, path) in sorted(layers[mask].items()):
                for nxt in adjacency[last]:
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    new_mask = mask | bit
                    candidate_value = current + float(matrix[last, nxt])
                    candidate_path = path + (nxt,)
                    previous = layers[new_mask].get(nxt)
                    if (
                        previous is None
                        or candidate_value < previous[0]
                        or (
                            candidate_value == previous[0]
                            and candidate_path < previous[1]
                        )
                    ):
                        layers[new_mask][nxt] = (
                            candidate_value,
                            candidate_path,
                        )

        for end in range(start + 1, n):
            solution = layers[full_mask].get(end)
            if solution is None:
                continue
            value, path = solution
            result[(start, end)] = EndpointHamiltonianPath(
                float(value),
                path,
            )

    return dict(sorted(result.items()))


class _DisjointSet:
    def __init__(self, vertices: Iterable[int]) -> None:
        self.parent = {int(vertex): int(vertex) for vertex in vertices}
        self.rank = {int(vertex): 0 for vertex in vertices}

    def find(self, vertex: int) -> int:
        parent = self.parent[vertex]
        if parent != vertex:
            self.parent[vertex] = self.find(parent)
        return self.parent[vertex]

    def union(self, u: int, v: int) -> bool:
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return False
        if self.rank[root_u] < self.rank[root_v]:
            root_u, root_v = root_v, root_u
        self.parent[root_v] = root_u
        if self.rank[root_u] == self.rank[root_v]:
            self.rank[root_u] += 1
        return True


def _sparse_mst_cost(
    dist: np.ndarray,
    vertices: Sequence[int],
    allowed_edges: set[Edge],
    forced_edge: Edge | None = None,
    forced_weight: float = 0.0,
) -> float:
    """Kruskal cost on an induced sparse graph, optionally forcing one edge."""

    active = tuple(sorted(map(int, vertices)))
    if len(active) <= 1:
        return 0.0 if forced_edge is None else inf
    active_set = set(active)
    dsu = _DisjointSet(active)
    edge_count = 0
    total = 0.0

    forced = None
    if forced_edge is not None:
        forced = (
            forced_edge
            if forced_edge[0] < forced_edge[1]
            else (forced_edge[1], forced_edge[0])
        )
        if not set(forced) <= active_set:
            return inf
        if not dsu.union(*forced):
            return inf
        total += float(forced_weight)
        edge_count += 1

    candidates = sorted(
        (
            (float(dist[u, v]), (u, v))
            for u, v in allowed_edges
            if u in active_set
            and v in active_set
            and (u, v) != forced
        ),
        key=lambda item: (item[0], item[1]),
    )
    for weight, candidate in candidates:
        if dsu.union(*candidate):
            total += weight
            edge_count += 1
            if edge_count == len(active) - 1:
                return float(total)
    return inf


def _sparse_mst_solution(
    weights: np.ndarray,
    vertices: Sequence[int],
    allowed_edges: set[Edge],
) -> tuple[float, tuple[Edge, ...]]:
    """Return a deterministic sparse MST under arbitrary symmetric weights."""

    active = tuple(sorted(map(int, vertices)))
    if len(active) <= 1:
        return 0.0, ()
    active_set = set(active)
    dsu = _DisjointSet(active)
    chosen: list[Edge] = []
    total = 0.0
    candidates = sorted(
        (
            (float(weights[u, v]), (u, v))
            for u, v in allowed_edges
            if u in active_set and v in active_set
        ),
        key=lambda item: (item[0], item[1]),
    )
    for weight, candidate in candidates:
        if dsu.union(*candidate):
            chosen.append(candidate)
            total += weight
            if len(chosen) == len(active) - 1:
                return float(total), tuple(chosen)
    return inf, ()


def _sparse_forced_mst_solution(
    weights: np.ndarray,
    vertices: Sequence[int],
    allowed_edges: set[Edge],
    forced_edge: Edge,
    forced_weight: float,
) -> tuple[float, tuple[Edge, ...]]:
    """Deterministic sparse MST constrained to contain one extra edge."""

    active = tuple(sorted(map(int, vertices)))
    active_set = set(active)
    forced = (
        forced_edge
        if forced_edge[0] < forced_edge[1]
        else (forced_edge[1], forced_edge[0])
    )
    if len(active) <= 1 or not set(forced) <= active_set:
        return inf, ()
    dsu = _DisjointSet(active)
    if not dsu.union(*forced):
        return inf, ()
    chosen: list[Edge] = [forced]
    total = float(forced_weight)
    candidates = sorted(
        (
            (float(weights[u, v]), (u, v))
            for u, v in allowed_edges
            if u in active_set and v in active_set and (u, v) != forced
        ),
        key=lambda item: (item[0], item[1]),
    )
    for weight, candidate in candidates:
        if dsu.union(*candidate):
            chosen.append(candidate)
            total += weight
            if len(chosen) == len(active) - 1:
                return float(total), tuple(chosen)
    return inf, ()


def _incident_costs(
    dist: np.ndarray,
    baseline_edges: set[Edge],
    root: int,
) -> list[float]:
    return sorted(
        float(dist[u, v])
        for u, v in baseline_edges
        if u == root or v == root
    )


def baseline_cycle_one_tree_lower_bound(
    dist: np.ndarray,
    baseline_edges: Iterable[Edge],
) -> float:
    """Maximum-root 1-tree lower bound on a baseline Hamiltonian cycle."""

    matrix = _validate_distance_matrix(dist)
    n = len(matrix)
    baseline = _normalize_allowed_edges(n, baseline_edges)
    bounds: list[float] = []
    for root in range(n):
        tree = _sparse_mst_cost(
            matrix,
            [vertex for vertex in range(n) if vertex != root],
            baseline,
        )
        incident = _incident_costs(matrix, baseline, root)
        if not np.isfinite(tree) or len(incident) < 2:
            return inf
        bounds.append(tree + incident[0] + incident[1])
    return float(max(bounds))


def _cycle_root_one_tree_solution(
    dist: np.ndarray,
    baseline_edges: set[Edge],
    root: int,
    potentials: np.ndarray,
) -> tuple[float, tuple[Edge, ...], tuple[int, ...]]:
    """Minimum root 1-tree for Lagrangian degree potentials."""

    modified = (
        dist
        + potentials[:, None]
        + potentials[None, :]
    )
    tree_cost, tree_edges = _sparse_mst_solution(
        modified,
        [vertex for vertex in range(len(dist)) if vertex != root],
        baseline_edges,
    )
    incident = sorted(
        (
            (float(modified[u, v]), (u, v))
            for u, v in baseline_edges
            if u == root or v == root
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not np.isfinite(tree_cost) or len(incident) < 2:
        return inf, (), ()
    root_edges = (incident[0][1], incident[1][1])
    selected = (*tree_edges, *root_edges)
    degrees = [0] * len(dist)
    for u, v in selected:
        degrees[u] += 1
        degrees[v] += 1
    modified_cost = tree_cost + incident[0][0] + incident[1][0]
    dual_value = float(modified_cost - 2.0 * float(potentials.sum()))
    return dual_value, tuple(selected), tuple(degrees)


def baseline_cycle_held_karp_lower_bound(
    dist: np.ndarray,
    baseline_edges: Iterable[Edge],
    upper_bound: float,
    iterations: int = 200,
) -> float:
    """Potential-optimized Held--Karp 1-tree lower bound.

    For every fixed root and every potential vector ``pi``, the minimum
    root-1-tree under modified costs ``c(i,j)+pi_i+pi_j``, minus
    ``2*sum(pi)``, is a valid Hamiltonian-cycle lower bound.  The deterministic
    subgradient schedule only searches for good potentials; validity does not
    depend on convergence.

    Integer-valued matrices use integer potentials, preserving exact binary64
    sums at the small scales guarded by the study runner.  Raw Euclidean
    matrices use real potentials and remain tolerance-audited numerically.
    """

    matrix = _validate_distance_matrix(dist)
    n = len(matrix)
    baseline = _normalize_allowed_edges(n, baseline_edges)
    if not np.isfinite(upper_bound) or upper_bound < 0.0:
        raise ValueError(
            "upper_bound must be a nonnegative finite cycle-witness cost"
        )
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    integral = bool(np.array_equal(matrix, np.rint(matrix)))
    best_over_roots = -inf

    for root in range(n):
        potentials = np.zeros(n, dtype=float)
        root_best = -inf
        scale = 2.0
        stagnant = 0
        for _ in range(iterations + 1):
            if integral:
                _guard_exact_integer_dual_arithmetic(matrix, potentials)
            value, _, degrees = _cycle_root_one_tree_solution(
                matrix,
                baseline,
                root,
                potentials,
            )
            if not np.isfinite(value):
                raise ArithmeticError(
                    "a finite cycle witness produced a nonfinite "
                    "Held--Karp relaxation"
                )
            if value > root_best:
                root_best = value
                stagnant = 0
            else:
                stagnant += 1
            subgradient = np.asarray(degrees, dtype=float) - 2.0
            norm_squared = float(subgradient @ subgradient)
            if norm_squared == 0.0:
                break
            gap = max(0.0, float(upper_bound - value))
            if gap == 0.0:
                break
            raw_step = scale * gap / norm_squared
            if integral:
                step = float(max(1, int(np.floor(raw_step))))
            else:
                step = raw_step
                if step <= np.finfo(float).eps:
                    break
            potentials = potentials + step * subgradient
            if integral:
                potentials = np.rint(potentials)
            # An exact gauge shift preserves the dual value and modified-edge
            # ordering.  Anchoring at the root also preserves integer
            # potentials without mean/rounding ambiguity.
            potentials = potentials - float(potentials[root])
            if integral:
                _guard_exact_integer_dual_arithmetic(matrix, potentials)
            if stagnant >= 20:
                scale *= 0.5
                stagnant = 0
        best_over_roots = max(best_over_roots, root_best)

    return float(best_over_roots)


def endpoint_path_forced_one_tree_lower_bound(
    dist: np.ndarray,
    baseline_edges: Iterable[Edge],
    endpoints: Edge,
) -> float:
    """Forced-zero-edge 1-tree lower bound on ``H_D(s,t)``.

    Add the nonbaseline endpoint edge with artificial cost zero.  Every
    baseline Hamiltonian ``s``--``t`` path then becomes a cycle of unchanged
    cost.  Removing each possible root yields the root-specific bounds below;
    their maximum remains a lower bound on the endpoint-path optimum.
    """

    matrix = _validate_distance_matrix(dist)
    n = len(matrix)
    baseline = _normalize_allowed_edges(n, baseline_edges)
    s, t = _normalize_endpoint_pair(n, endpoints)
    if (s, t) in baseline:
        raise ValueError("the artificial endpoint edge must be nonbaseline")

    bounds: list[float] = []
    for root in range(n):
        active = [vertex for vertex in range(n) if vertex != root]
        incident = _incident_costs(matrix, baseline, root)
        if root in (s, t):
            tree = _sparse_mst_cost(matrix, active, baseline)
            if not np.isfinite(tree) or not incident:
                return inf
            bounds.append(tree + incident[0])
        else:
            tree = _sparse_mst_cost(
                matrix,
                active,
                baseline,
                forced_edge=(s, t),
                forced_weight=0.0,
            )
            if not np.isfinite(tree) or len(incident) < 2:
                return inf
            bounds.append(tree + incident[0] + incident[1])
    return float(max(bounds))


def _path_root_one_tree_solution(
    dist: np.ndarray,
    baseline_edges: set[Edge],
    endpoints: Edge,
    root: int,
    potentials: np.ndarray,
) -> tuple[float, tuple[Edge, ...], tuple[int, ...]]:
    """Minimum forced-endpoint root 1-tree under degree potentials."""

    n = len(dist)
    s, t = endpoints
    modified = (
        dist
        + potentials[:, None]
        + potentials[None, :]
    )
    artificial_weight = float(potentials[s] + potentials[t])
    active = [vertex for vertex in range(n) if vertex != root]
    incident = sorted(
        (
            (float(modified[u, v]), (u, v))
            for u, v in baseline_edges
            if u == root or v == root
        ),
        key=lambda item: (item[0], item[1]),
    )

    if root in (s, t):
        tree_cost, tree_edges = _sparse_mst_solution(
            modified,
            active,
            baseline_edges,
        )
        if not np.isfinite(tree_cost) or not incident:
            return inf, (), ()
        root_edges = (incident[0][1],)
        selected = (*tree_edges, *root_edges, (s, t))
        modified_cost = tree_cost + incident[0][0] + artificial_weight
    else:
        tree_cost, tree_edges = _sparse_forced_mst_solution(
            modified,
            active,
            baseline_edges,
            forced_edge=(s, t),
            forced_weight=artificial_weight,
        )
        if not np.isfinite(tree_cost) or len(incident) < 2:
            return inf, (), ()
        root_edges = (incident[0][1], incident[1][1])
        selected = (*tree_edges, *root_edges)
        modified_cost = tree_cost + incident[0][0] + incident[1][0]

    degrees = [0] * n
    for u, v in selected:
        degrees[u] += 1
        degrees[v] += 1
    dual_value = float(modified_cost - 2.0 * float(potentials.sum()))
    return dual_value, tuple(selected), tuple(degrees)


def endpoint_path_held_karp_lower_bound(
    dist: np.ndarray,
    baseline_edges: Iterable[Edge],
    endpoints: Edge,
    upper_bound: float,
    iterations: int = 100,
) -> float:
    """Potential-optimized forced-edge 1-tree lower bound on ``H_D``."""

    matrix = _validate_distance_matrix(dist)
    n = len(matrix)
    baseline = _normalize_allowed_edges(n, baseline_edges)
    s, t = _normalize_endpoint_pair(n, endpoints)
    if (s, t) in baseline:
        raise ValueError("endpoints must be a nonbaseline pair")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if np.isnan(upper_bound) or upper_bound == -inf or upper_bound < 0.0:
        raise ValueError(
            "upper_bound must be nonnegative finite or positive infinity"
        )
    if upper_bound == inf:
        return endpoint_path_forced_one_tree_lower_bound(
            matrix,
            baseline,
            (s, t),
        )

    integral = bool(np.array_equal(matrix, np.rint(matrix)))
    best_over_roots = -inf
    for root in range(n):
        potentials = np.zeros(n, dtype=float)
        root_best = -inf
        scale = 2.0
        stagnant = 0
        for _ in range(iterations + 1):
            if integral:
                _guard_exact_integer_dual_arithmetic(matrix, potentials)
            value, _, degrees = _path_root_one_tree_solution(
                matrix,
                baseline,
                (s, t),
                root,
                potentials,
            )
            if not np.isfinite(value):
                raise ArithmeticError(
                    "a finite endpoint-path witness produced a nonfinite "
                    "forced-edge Held--Karp relaxation"
                )
            if value > root_best:
                root_best = value
                stagnant = 0
            else:
                stagnant += 1
            subgradient = np.asarray(degrees, dtype=float) - 2.0
            norm_squared = float(subgradient @ subgradient)
            if norm_squared == 0.0:
                break
            gap = max(0.0, float(upper_bound - value))
            if gap == 0.0:
                break
            raw_step = scale * gap / norm_squared
            if integral:
                step = float(max(1, int(np.floor(raw_step))))
            else:
                step = raw_step
                if step <= np.finfo(float).eps:
                    break
            potentials = potentials + step * subgradient
            if integral:
                potentials = np.rint(potentials)
            potentials = potentials - float(potentials[root])
            if integral:
                _guard_exact_integer_dual_arithmetic(matrix, potentials)
            if stagnant >= 20:
                scale *= 0.5
                stagnant = 0
        best_over_roots = max(best_over_roots, root_best)
    return float(best_over_roots)


def analyze_pair_closure_thresholds(
    dist: np.ndarray,
    baseline_edges: Iterable[Edge],
    held_karp_bound_iterations: int = 200,
    held_karp_path_bound_iterations: int = 40,
) -> PairClosureThresholdAnalysis:
    """Compute exact-small q=1 thresholds and 1-tree sandwiches.

    This exact-small routine deliberately separates *truth* from the proof
    certificates.  Exact Held--Karp values calibrate the experiment, while the
    inequalities use only the validity of the returned feasible witnesses and
    the independently computed 1-tree lower bounds.
    """

    # Imported lazily to keep the endpoint-path primitive independent of the
    # cycle solver and avoid a module-level cycle.
    from .exact import held_karp_cycle

    matrix = _validate_distance_matrix(dist)
    n = len(matrix)
    baseline = _normalize_allowed_edges(n, baseline_edges)
    cycle_exact, cycle_path = held_karp_cycle(
        matrix,
        allowed_edges=baseline,
    )
    if not np.isfinite(cycle_exact) or not cycle_path:
        raise ValueError(
            "metric pair-threshold analysis requires a baseline "
            "Hamiltonian cycle"
        )
    cycle_one_tree_lower = baseline_cycle_one_tree_lower_bound(
        matrix,
        baseline,
    )
    cycle_held_karp_lower = baseline_cycle_held_karp_lower_bound(
        matrix,
        baseline,
        upper_bound=float(cycle_exact),
        iterations=int(held_karp_bound_iterations),
    )
    cycle_lower = max(cycle_one_tree_lower, cycle_held_karp_lower)
    if not np.isfinite(cycle_lower):
        raise AssertionError(
            "a baseline cycle witness must make every root 1-tree feasible"
        )

    endpoint_paths = held_karp_all_endpoint_paths(matrix, baseline)
    complement = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if (u, v) not in baseline
    ]
    pairs: list[PairClosureThreshold] = []
    for candidate in complement:
        endpoint = endpoint_paths[candidate]
        edge_cost = float(matrix[candidate])
        path_one_tree_lower = endpoint_path_forced_one_tree_lower_bound(
            matrix,
            baseline,
            candidate,
        )
        path_upper = float(endpoint.value)
        path_held_karp_lower = endpoint_path_held_karp_lower_bound(
            matrix,
            baseline,
            candidate,
            upper_bound=path_upper,
            iterations=int(held_karp_path_bound_iterations),
        )
        path_lower = max(path_one_tree_lower, path_held_karp_lower)
        closure_exact = path_upper + edge_cost
        release_exact = float(cycle_exact - path_upper)
        gain_exact = float(release_exact - edge_cost)
        gain_lower = float(cycle_lower - path_upper - edge_cost)
        gain_upper = float(cycle_exact - path_lower - edge_cost)
        if edge_cost > 0.0:
            kappa_exact = float(release_exact / edge_cost)
            kappa_lower = float((cycle_lower - path_upper) / edge_cost)
            kappa_upper = float((cycle_exact - path_lower) / edge_cost)
        else:
            kappa_exact = np.nan
            kappa_lower = np.nan
            kappa_upper = np.nan
        pairs.append(
            PairClosureThreshold(
                edge=candidate,
                edge_cost=edge_cost,
                path_exact=path_upper,
                path_witness=endpoint.path,
                closure_exact=float(closure_exact),
                cycle_lower=float(cycle_lower),
                cycle_one_tree_lower=float(cycle_one_tree_lower),
                cycle_held_karp_lower=float(cycle_held_karp_lower),
                cycle_upper=float(cycle_exact),
                path_lower=float(path_lower),
                path_one_tree_lower=float(path_one_tree_lower),
                path_held_karp_lower=float(path_held_karp_lower),
                path_upper=path_upper,
                release_exact=release_exact,
                gain_exact=gain_exact,
                gain_lower=gain_lower,
                gain_upper=gain_upper,
                kappa_exact=kappa_exact,
                kappa_lower=kappa_lower,
                kappa_upper=kappa_upper,
            )
        )

    return PairClosureThresholdAnalysis(
        cycle_exact=float(cycle_exact),
        cycle_witness=tuple(map(int, cycle_path)),
        cycle_lower=float(cycle_lower),
        cycle_one_tree_lower=float(cycle_one_tree_lower),
        cycle_held_karp_lower=float(cycle_held_karp_lower),
        pairs=tuple(pairs),
    )
