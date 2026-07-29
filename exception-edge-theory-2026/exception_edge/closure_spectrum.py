from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf
from operator import index
from typing import Iterable, Sequence

import numpy as np

from .exact import held_karp_cycle_by_exception_count
from .geometry import Edge, edge, tour_edges


@dataclass(frozen=True)
class ClosureDecomposition:
    """Baseline/closure decomposition of one Hamiltonian cycle."""

    exception_count: int
    kind: str
    baseline_edges: frozenset[Edge]
    outside_edges: frozenset[Edge]
    component_sizes: tuple[int, ...]
    singleton_component_count: int


@dataclass(frozen=True)
class ClosureSpectrumLevel:
    """One feasible exact-exception-count layer."""

    q: int
    value: float
    tour: tuple[int, ...]
    decomposition: ClosureDecomposition
    budget_envelope: float


@dataclass(frozen=True)
class ClosureSpectrum:
    """Exact baseline-relative Hamiltonian closure spectrum."""

    levels: tuple[ClosureSpectrumLevel, ...]
    optimum: float
    numeric_argmin_q: tuple[int, ...]
    optimal_q_within_tolerance: tuple[int, ...]
    minimum_feasible_q: int
    optimal_q_min: int
    optimal_q_max: int
    metric_surplus_min: int
    classification: str
    tolerance_absolute: float

    def level(self, q: int) -> ClosureSpectrumLevel | None:
        return next((current for current in self.levels if current.q == q), None)

    @property
    def optimal_q(self) -> tuple[int, ...]:
        """Compatibility alias for the operational tolerance set.

        The mathematical set ``Q*={q: Z_q=z*}`` is distinct from this
        floating-point decision set.  For an exactly represented integer
        objective and a tolerance smaller than half an objective unit they
        coincide; raw Euclidean output must be read as ``Qhat*_tau``.
        """

        return self.optimal_q_within_tolerance


def _validate_n(n: int) -> int:
    try:
        normalized = index(n)
    except TypeError as exc:
        raise ValueError("n must be an integer") from exc
    if normalized < 3:
        raise ValueError("Hamiltonian cycle requires n >= 3")
    return normalized


def _normalize_baseline(n: int, baseline_edges: Iterable[Edge]) -> set[Edge]:
    n = _validate_n(n)
    result: set[Edge] = set()
    try:
        iterator = iter(baseline_edges)
    except TypeError as exc:
        raise ValueError("baseline_edges must be an iterable of edge pairs") from exc
    for candidate in iterator:
        try:
            raw_u, raw_v = candidate
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each baseline edge must contain exactly two endpoints"
            ) from exc
        try:
            u, v = index(raw_u), index(raw_v)
        except TypeError as exc:
            raise ValueError("baseline edge endpoints must be integers") from exc
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"baseline edge {(u, v)} is outside 0..{n - 1}")
        result.add(edge(u, v))
    return result


def _validate_tour(tour: Sequence[int], n: int) -> tuple[int, ...]:
    try:
        normalized = tuple(index(vertex) for vertex in tour)
    except TypeError as exc:
        raise ValueError("tour vertices must be integers") from exc
    if len(normalized) != n or set(normalized) != set(range(n)):
        raise ValueError("tour must contain every vertex exactly once")
    return normalized


def _components(
    n: int,
    edges: Iterable[Edge],
    active_vertices: Iterable[int] | None = None,
) -> list[tuple[int, ...]]:
    active = (
        set(range(n))
        if active_vertices is None
        else set(map(int, active_vertices))
    )
    if not active:
        return []
    adjacency = {vertex: [] for vertex in active}
    for u, v in edges:
        if u in active and v in active:
            adjacency[u].append(v)
            adjacency[v].append(u)

    result: list[tuple[int, ...]] = []
    unseen = set(active)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        current: list[int] = []
        while stack:
            vertex = stack.pop()
            current.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(current)))
    return sorted(result, key=lambda values: (values[0], len(values), values))


def decompose_tour(
    tour: Sequence[int],
    baseline_edges: Iterable[Edge],
) -> ClosureDecomposition:
    """Split a tour into its baseline path cover and outside closure.

    For q=0 the baseline part is a Hamiltonian cycle and is deliberately not
    described as a path cover.  For q>0, all fixed-tour closure invariants are
    checked, including singleton components created by adjacent exceptions.
    """

    n = len(tour)
    if n < 3:
        raise ValueError("Hamiltonian cycle requires n >= 3")
    normalized_tour = _validate_tour(tour, n)
    baseline = _normalize_baseline(n, baseline_edges)
    cycle = tour_edges(normalized_tour)
    inside = cycle & baseline
    outside = cycle - baseline
    q = len(outside)

    if q == 0:
        if len(inside) != n:
            raise AssertionError("baseline cycle must contain n distinct edges")
        return ClosureDecomposition(
            exception_count=0,
            kind="baseline_cycle",
            baseline_edges=frozenset(inside),
            outside_edges=frozenset(),
            component_sizes=(),
            singleton_component_count=0,
        )

    if len(inside) != n - q:
        raise AssertionError("removing q cycle edges must leave n-q edges")

    degrees_inside = [0] * n
    degrees_outside = [0] * n
    for u, v in inside:
        degrees_inside[u] += 1
        degrees_inside[v] += 1
    for u, v in outside:
        degrees_outside[u] += 1
        degrees_outside[v] += 1

    if any(value > 2 for value in degrees_inside):
        raise AssertionError("baseline remainder is not a linear forest")
    if any(
        degrees_inside[v] + degrees_outside[v] != 2 for v in range(n)
    ):
        raise AssertionError("closure does not restore degree two")
    if sum(2 - value for value in degrees_inside) != 2 * q:
        raise AssertionError("endpoint-stub count must equal 2q")

    components = _components(n, inside)
    if len(components) != q:
        raise AssertionError("q removed cycle edges must create q components")
    for component in components:
        component_set = set(component)
        induced_edge_count = sum(
            1 for u, v in inside if u in component_set and v in component_set
        )
        if induced_edge_count != len(component) - 1:
            raise AssertionError("baseline remainder contains a cycle")

    sizes = tuple(sorted(map(len, components)))
    return ClosureDecomposition(
        exception_count=q,
        kind="linear_forest_closure",
        baseline_edges=frozenset(inside),
        outside_edges=frozenset(outside),
        component_sizes=sizes,
        singleton_component_count=sum(size == 1 for size in sizes),
    )


def degree_deficit_lower_bound(n: int, baseline_edges: Iterable[Edge]) -> int:
    """Safe lower bound from missing degree-two incidences."""

    n = _validate_n(n)
    baseline = _normalize_baseline(n, baseline_edges)
    degrees = [0] * n
    for u, v in baseline:
        degrees[u] += 1
        degrees[v] += 1
    return int(ceil(sum(max(0, 2 - value) for value in degrees) / 2.0))


def component_lower_bound(n: int, baseline_edges: Iterable[Edge]) -> int:
    """Safe bound for a disconnected baseline (zero when connected)."""

    n = _validate_n(n)
    baseline = _normalize_baseline(n, baseline_edges)
    count = len(_components(n, baseline))
    return int(count if count > 1 else 0)


def cut_deficit_lower_bound(n: int, baseline_edges: Iterable[Edge]) -> int:
    """Maximum single-cut deficit max_S (2-|delta_G0(S)|)_+."""

    n = _validate_n(n)
    baseline = _normalize_baseline(n, baseline_edges)
    full = (1 << n) - 1
    best = 0
    for mask in range(1, full):
        if not (mask & 1):
            continue  # identify S and its complement
        crossing = sum(
            ((mask >> u) & 1) != ((mask >> v) & 1)
            for u, v in baseline
        )
        best = max(best, 2 - crossing)
    return int(max(0, best))


def toughness_deficit_lower_bound(
    n: int,
    baseline_edges: Iterable[Edge],
) -> tuple[int, tuple[int, ...]]:
    """Exact small-n toughness-deficit bound and one maximizing set.

    The maximum is over nonempty proper S.  S=empty is excluded because the
    Hamiltonian-implies-1-tough argument does not apply there.
    """

    n = _validate_n(n)
    baseline = _normalize_baseline(n, baseline_edges)
    full = (1 << n) - 1
    best = 0
    witness: tuple[int, ...] = (0,)
    for mask in range(1, full):
        removed = tuple(v for v in range(n) if mask & (1 << v))
        active = [v for v in range(n) if not (mask & (1 << v))]
        deficiency = max(0, len(_components(n, baseline, active)) - len(removed))
        if deficiency > best:
            best = deficiency
            witness = removed
    return int(best), witness


def analyze_closure_spectrum(
    dist: np.ndarray,
    baseline_edges: Iterable[Edge],
    tolerance: float = 1e-10,
) -> ClosureSpectrum:
    """Compute Z_q, its budget envelope, and topology/metric classification."""

    dist = np.asarray(dist, dtype=float)
    n = len(dist)
    if dist.shape != (n, n):
        raise ValueError("dist must be a square matrix")
    _validate_n(n)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    baseline = _normalize_baseline(n, baseline_edges)
    raw = held_karp_cycle_by_exception_count(dist, baseline)
    if not raw:
        raise AssertionError("complete ambient graph must contain a tour")

    optimum = min(float(value) for value, _ in raw.values())
    tolerance_absolute = tolerance * max(1.0, abs(optimum))
    numeric_argmin_q = tuple(
        sorted(q for q, (value, _) in raw.items() if float(value) == optimum)
    )
    optimal_q_within_tolerance = tuple(
        sorted(
            q
            for q, (value, _) in raw.items()
            if abs(float(value) - optimum) <= tolerance_absolute
        )
    )
    if not numeric_argmin_q or not optimal_q_within_tolerance:
        raise AssertionError("at least one spectrum level must be optimal")

    levels: list[ClosureSpectrumLevel] = []
    envelope = inf
    for q in sorted(raw):
        value, tour = raw[q]
        envelope = min(envelope, float(value))
        decomposition = decompose_tour(tour, baseline)
        if decomposition.exception_count != q:
            raise AssertionError("witness tour does not match its q layer")
        levels.append(
            ClosureSpectrumLevel(
                q=int(q),
                value=float(value),
                tour=tuple(map(int, tour)),
                decomposition=decomposition,
                budget_envelope=float(envelope),
            )
        )

    minimum_feasible_q = min(raw)
    optimal_q_min = min(optimal_q_within_tolerance)
    optimal_q_max = max(optimal_q_within_tolerance)
    metric_surplus = optimal_q_min - minimum_feasible_q

    if 0 not in raw:
        classification = (
            "topological_plus_metric"
            if metric_surplus > 0
            else "topological_required"
        )
    elif 0 in optimal_q_within_tolerance and any(
        q > 0 for q in optimal_q_within_tolerance
    ):
        classification = "baseline_exception_count_tie"
    elif optimal_q_within_tolerance == (0,):
        classification = "baseline_only_optimal_count"
    else:
        classification = "metric_exception_required"

    return ClosureSpectrum(
        levels=tuple(levels),
        optimum=float(optimum),
        numeric_argmin_q=numeric_argmin_q,
        optimal_q_within_tolerance=optimal_q_within_tolerance,
        minimum_feasible_q=int(minimum_feasible_q),
        optimal_q_min=int(optimal_q_min),
        optimal_q_max=int(optimal_q_max),
        metric_surplus_min=int(metric_surplus),
        classification=classification,
        tolerance_absolute=float(tolerance_absolute),
    )
