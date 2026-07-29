"""Iterative MILP solver for an undirected Hamiltonian cycle.

The initial model is the minimum-cost undirected 2-factor problem.  Whenever
its incumbent contains more than one connected component, a subtour
elimination constraint is added for every component and the model is solved
again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from operator import index
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_array

Edge = tuple[int, int]


@dataclass(frozen=True)
class SubtourMILPRound:
    """One solver round before newly found SECs are appended."""

    iteration: int
    solver_status: int
    objective: float | None
    lower_bound: float | None
    component_sizes: tuple[int, ...]
    cuts_before: int
    cuts_added: int
    mip_node_count: int | None = None
    mip_gap: float | None = None
    wall_seconds: float = 0.0


@dataclass(frozen=True)
class SubtourMILPResult:
    """Result and audit trail from an iterative subtour-elimination solve."""

    status: str
    objective: float | None
    lower_bound: float | None
    cycle: tuple[int, ...] | None
    selected_edges: tuple[Edge, ...]
    iterations: int
    cuts: int
    sec_subsets: tuple[tuple[int, ...], ...]
    rounds: tuple[SubtourMILPRound, ...]
    exact: bool
    solver_status: int | None
    message: str
    wall_seconds: float = 0.0
    first_feasible_wall_seconds_upper_bound: float | None = None
    best_feasible_wall_seconds_upper_bound: float | None = None


def _components(n_vertices: int, selected_edges: tuple[Edge, ...]) -> list[tuple[int, ...]]:
    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for u, v in selected_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    remaining = set(range(n_vertices))
    result: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: list[int] = []
        remaining.remove(start)
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency[u]:
                if v in remaining:
                    remaining.remove(v)
                    stack.append(v)
        result.append(tuple(sorted(component)))
    return sorted(result)


def _cycle_from_edges(
    n_vertices: int, selected_edges: tuple[Edge, ...]
) -> tuple[int, ...] | None:
    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for u, v in selected_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    if any(len(neighbors) != 2 for neighbors in adjacency):
        return None

    # Start at vertex zero and select the lexicographically smaller orientation.
    cycle = [0]
    previous = -1
    current = 0
    first_neighbor = min(adjacency[0])
    for _ in range(1, n_vertices):
        neighbors = adjacency[current]
        nxt = first_neighbor if current == 0 else (
            neighbors[0] if neighbors[0] != previous else neighbors[1]
        )
        if nxt == 0 or nxt in cycle:
            return None
        cycle.append(nxt)
        previous, current = current, nxt
    if 0 not in adjacency[current]:
        return None
    return tuple(cycle)


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if isfinite(converted) else None


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric):
        return None
    converted = int(numeric)
    return converted if converted >= 0 else None


def _canonical_sec_subset(
    component: tuple[int, ...],
    n_vertices: int,
) -> tuple[int, ...] | None:
    """Identify complementary SECs and omit degree-implied singleton cuts."""
    if not 1 < len(component) < n_vertices - 1:
        return None
    complement = tuple(sorted(set(range(n_vertices)) - set(component)))
    return min(component, complement, key=lambda subset: (len(subset), subset))


def solve_hamiltonian_cycle(
    n_vertices: int,
    edge_costs: Mapping[tuple[int, int], float],
    *,
    time_limit: float | None = None,
    total_time_limit: float | None = None,
    max_iterations: int = 100,
    objective_upper_bound: float | None = None,
    initial_sec_subsets: Iterable[Iterable[int]] | None = None,
    continue_after_feasible_limit: bool = False,
    objective_granularity: float | None = None,
    threads: int | None = None,
    random_seed: int | None = None,
) -> SubtourMILPResult:
    """Solve a weighted Hamiltonian-cycle problem on an undirected graph.

    Vertices are ``range(n_vertices)`` and ``edge_costs`` maps undirected edge
    pairs to finite costs.  Endpoint order is immaterial, but supplying both
    orientations of the same edge is rejected.  ``time_limit`` is passed to
    every individual SciPy MILP solve, rather than shared across cut rounds.
    ``total_time_limit`` is a wall-clock deadline for the entire call.  When
    both are supplied, each round receives the smaller of its per-round limit
    and the remaining total time.

    ``objective_upper_bound`` adds ``sum(cost[e] * x[e]) <= bound``.  It is
    useful when a known feasible tour value is available.  A timeout may
    return a feasible cycle, but ``exact`` is true only when the final SciPy
    solve reports optimality and that solution is one connected cycle.

    ``initial_sec_subsets`` may contain valid SEC vertex subsets carried from
    a previous solve on the same vertex set.  This is useful when a candidate
    graph is expanded: SECs remain valid after adding edges even though SciPy
    does not expose a MIP-start interface.

    If ``continue_after_feasible_limit`` is true, a connected incumbent found
    on a limit exit is retained and the search continues with a strict
    objective cutoff.  ``objective_granularity`` must then be a positive unit
    dividing every edge cost; it lets an infeasible ``incumbent - unit`` model
    certify the retained tour as optimal on the supplied graph.

    ``threads`` and ``random_seed`` are forwarded to HiGHS.  For reproducible
    timed comparisons, use a fresh process and set ``threads=1`` before the
    first MILP call so HiGHS initializes its global scheduler consistently.
    """
    total_started = time.perf_counter()

    try:
        n = index(n_vertices)
    except TypeError as exc:
        raise ValueError("n_vertices must be an integer") from exc
    if n < 3:
        raise ValueError("a Hamiltonian cycle requires at least three vertices")
    try:
        iteration_limit = index(max_iterations)
    except TypeError as exc:
        raise ValueError("max_iterations must be an integer") from exc
    if iteration_limit < 1:
        raise ValueError("max_iterations must be positive")
    if time_limit is not None:
        time_limit = float(time_limit)
        if not isfinite(time_limit) or time_limit <= 0.0:
            raise ValueError("time_limit must be finite and positive")
    if total_time_limit is not None:
        total_time_limit = float(total_time_limit)
        if not isfinite(total_time_limit) or total_time_limit <= 0.0:
            raise ValueError("total_time_limit must be finite and positive")
    if objective_upper_bound is not None:
        objective_upper_bound = float(objective_upper_bound)
        if not isfinite(objective_upper_bound):
            raise ValueError("objective_upper_bound must be finite")
    if not isinstance(continue_after_feasible_limit, bool):
        raise ValueError("continue_after_feasible_limit must be boolean")
    if objective_granularity is not None:
        objective_granularity = float(objective_granularity)
        if (
            not isfinite(objective_granularity)
            or objective_granularity <= 0.0
        ):
            raise ValueError("objective_granularity must be finite and positive")
    if continue_after_feasible_limit and objective_granularity is None:
        raise ValueError(
            "objective_granularity is required when continuing feasible limits"
        )
    if threads is not None:
        try:
            threads = index(threads)
        except TypeError as exc:
            raise ValueError("threads must be an integer") from exc
        if threads < 1:
            raise ValueError("threads must be positive")
    if random_seed is not None:
        try:
            random_seed = index(random_seed)
        except TypeError as exc:
            raise ValueError("random_seed must be an integer") from exc
        if random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
    if threads is not None:
        try:
            from scipy.optimize._highspy._core import _Highs

            _Highs.resetGlobalScheduler(True)
        except (ImportError, AttributeError, TypeError):
            # The option is still forwarded below.  A fresh benchmark
            # subprocess is the portable way to guarantee first-use setup.
            pass
    if not isinstance(edge_costs, Mapping):
        raise ValueError("edge_costs must be a mapping of edge pairs to costs")

    normalized: dict[Edge, float] = {}
    for candidate, raw_cost in edge_costs.items():
        try:
            raw_u, raw_v = candidate
        except (TypeError, ValueError) as exc:
            raise ValueError("each edge must contain exactly two endpoints") from exc
        try:
            u, v = index(raw_u), index(raw_v)
        except TypeError as exc:
            raise ValueError("edge endpoints must be integers") from exc
        if u == v:
            raise ValueError("self-loops are not valid edges")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge endpoint is outside range(n_vertices)")
        edge = (u, v) if u < v else (v, u)
        if edge in normalized:
            raise ValueError(f"duplicate undirected edge: {edge}")
        cost = float(raw_cost)
        if not isfinite(cost):
            raise ValueError("edge costs must be finite")
        normalized[edge] = cost
    if objective_granularity is not None:
        for candidate, cost in normalized.items():
            units = cost / objective_granularity
            if not np.isclose(units, round(units), atol=1e-9, rtol=0.0):
                raise ValueError(
                    f"edge cost for {candidate} is not a multiple of "
                    "objective_granularity"
                )

    edges = tuple(sorted(normalized))
    costs = np.asarray([normalized[edge] for edge in edges], dtype=float)
    m = len(edges)
    if m == 0:
        return SubtourMILPResult(
            status="infeasible",
            objective=None,
            lower_bound=None,
            cycle=None,
            selected_edges=(),
            iterations=0,
            cuts=0,
            sec_subsets=(),
            rounds=(),
            exact=False,
            solver_status=2,
            message="graph has no edges",
            wall_seconds=time.perf_counter() - total_started,
        )

    degree_rows: list[int] = []
    degree_columns: list[int] = []
    for column, (u, v) in enumerate(edges):
        degree_rows.extend((u, v))
        degree_columns.extend((column, column))
    degree_matrix = csr_array(
        (
            np.ones(len(degree_rows), dtype=float),
            (degree_rows, degree_columns),
        ),
        shape=(n, m),
    )
    degree_constraint = LinearConstraint(
        degree_matrix, np.full(n, 2.0), np.full(n, 2.0)
    )

    sec_subsets: list[tuple[int, ...]] = []
    seen_subsets: set[tuple[int, ...]] = set()
    if initial_sec_subsets is not None:
        for raw_subset in initial_sec_subsets:
            try:
                raw_vertices = tuple(index(vertex) for vertex in raw_subset)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "initial SEC subsets must be iterables of integer vertices"
                ) from exc
            if len(set(raw_vertices)) != len(raw_vertices):
                raise ValueError("an initial SEC subset contains duplicate vertices")
            if any(vertex < 0 or vertex >= n for vertex in raw_vertices):
                raise ValueError("an initial SEC vertex is outside range(n_vertices)")
            subset = _canonical_sec_subset(tuple(sorted(raw_vertices)), n)
            if subset is None:
                raise ValueError(
                    "initial SEC subsets must have between 2 and n-2 vertices"
                )
            if subset not in seen_subsets:
                seen_subsets.add(subset)
                sec_subsets.append(subset)
        sec_subsets.sort(key=lambda subset: (len(subset), subset))
    last_selected: tuple[Edge, ...] = ()
    last_objective: float | None = None
    last_lower_bound: float | None = None
    last_solver_status: int | None = None
    last_message = ""
    best_lower_bound: float | None = None
    best_cycle: tuple[int, ...] | None = None
    best_cycle_edges: tuple[Edge, ...] = ()
    best_cycle_objective: float | None = None
    first_feasible_wall_seconds_upper_bound: float | None = None
    best_feasible_wall_seconds_upper_bound: float | None = None
    round_records: list[SubtourMILPRound] = []

    for iteration in range(1, iteration_limit + 1):
        constraints: list[LinearConstraint] = [degree_constraint]
        active_objective_upper_bound = objective_upper_bound
        if (
            continue_after_feasible_limit
            and best_cycle_objective is not None
            and objective_granularity is not None
        ):
            strict_cutoff = best_cycle_objective - objective_granularity
            active_objective_upper_bound = (
                strict_cutoff
                if active_objective_upper_bound is None
                else min(active_objective_upper_bound, strict_cutoff)
            )
        if active_objective_upper_bound is not None:
            constraints.append(
                LinearConstraint(
                    csr_array(costs.reshape(1, -1)),
                    np.asarray([-np.inf]),
                    np.asarray([active_objective_upper_bound]),
                )
            )
        if sec_subsets:
            sec_rows: list[int] = []
            sec_columns: list[int] = []
            for row, subset in enumerate(sec_subsets):
                vertices = set(subset)
                for column, (u, v) in enumerate(edges):
                    if u in vertices and v in vertices:
                        sec_rows.append(row)
                        sec_columns.append(column)
            sec_matrix = csr_array(
                (
                    np.ones(len(sec_rows), dtype=float),
                    (sec_rows, sec_columns),
                ),
                shape=(len(sec_subsets), m),
            )
            constraints.append(
                LinearConstraint(
                    sec_matrix,
                    np.full(len(sec_subsets), -np.inf),
                    np.asarray([len(subset) - 1 for subset in sec_subsets], dtype=float),
                )
            )

        # A positive default MIP gap would make ``status == 0`` mean
        # "optimal within tolerance", which is too weak for an audit trail.
        options = {"disp": False, "mip_rel_gap": 0.0, "presolve": True}
        if threads is not None:
            options["threads"] = threads
        if random_seed is not None:
            options["random_seed"] = random_seed
        effective_time_limit = time_limit
        if total_time_limit is not None:
            remaining = total_time_limit - (time.perf_counter() - total_started)
            if remaining <= 0.0:
                if best_cycle is not None:
                    return SubtourMILPResult(
                        status="feasible_limit",
                        objective=best_cycle_objective,
                        lower_bound=best_lower_bound,
                        cycle=best_cycle,
                        selected_edges=best_cycle_edges,
                        iterations=len(round_records),
                        cuts=len(sec_subsets),
                        sec_subsets=tuple(sec_subsets),
                        rounds=tuple(round_records),
                        exact=False,
                        solver_status=1,
                        message=(
                            "total wall-clock limit reached after retaining "
                            "a connected incumbent"
                        ),
                        wall_seconds=time.perf_counter() - total_started,
                        first_feasible_wall_seconds_upper_bound=(
                            first_feasible_wall_seconds_upper_bound
                        ),
                        best_feasible_wall_seconds_upper_bound=(
                            best_feasible_wall_seconds_upper_bound
                        ),
                    )
                return SubtourMILPResult(
                    status="solver_limit",
                    objective=last_objective,
                    lower_bound=best_lower_bound,
                    cycle=None,
                    selected_edges=last_selected,
                    iterations=len(round_records),
                    cuts=len(sec_subsets),
                    sec_subsets=tuple(sec_subsets),
                    rounds=tuple(round_records),
                    exact=False,
                    solver_status=1,
                    message="total wall-clock limit reached before next MILP round",
                    wall_seconds=time.perf_counter() - total_started,
                )
            effective_time_limit = (
                remaining
                if effective_time_limit is None
                else min(effective_time_limit, remaining)
            )
        if effective_time_limit is not None:
            options["time_limit"] = effective_time_limit
        round_started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Unrecognized options detected: .*These will be passed "
                    "to HiGHS verbatim.*"
                ),
                category=RuntimeWarning,
            )
            result = milp(
                costs,
                integrality=np.ones(m, dtype=np.int8),
                bounds=Bounds(np.zeros(m), np.ones(m)),
                constraints=constraints,
                options=options,
            )
        round_wall_seconds = time.perf_counter() - round_started

        last_solver_status = int(result.status)
        last_message = str(result.message)
        last_objective = _optional_finite_float(getattr(result, "fun", None))
        # HiGHS supplies this for limit exits as well as optimal exits.  Keep
        # it even when no incumbent cycle is available.
        last_lower_bound = _optional_finite_float(
            getattr(result, "mip_dual_bound", None)
        )
        round_node_count = _optional_nonnegative_int(
            getattr(result, "mip_node_count", None)
        )
        round_mip_gap = _optional_finite_float(
            getattr(result, "mip_gap", None)
        )
        strict_cutoff_active = (
            continue_after_feasible_limit
            and best_cycle is not None
            and best_cycle_objective is not None
            and objective_granularity is not None
        )
        if (
            last_lower_bound is not None
            and (result.x is not None or not strict_cutoff_active)
            and (
                best_lower_bound is None
                or last_lower_bound > best_lower_bound
            )
        ):
            best_lower_bound = last_lower_bound
        if result.x is None:
            round_records.append(
                SubtourMILPRound(
                    iteration=iteration,
                    solver_status=last_solver_status,
                    objective=last_objective,
                    lower_bound=last_lower_bound,
                    component_sizes=(),
                    cuts_before=len(sec_subsets),
                    cuts_added=0,
                    mip_node_count=round_node_count,
                    mip_gap=round_mip_gap,
                    wall_seconds=round_wall_seconds,
                )
            )
            status = {
                1: "solver_limit",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(last_solver_status, "solver_error")
            if strict_cutoff_active and last_solver_status == 2:
                return SubtourMILPResult(
                    status="optimal_cutoff",
                    objective=best_cycle_objective,
                    lower_bound=best_cycle_objective,
                    cycle=best_cycle,
                    selected_edges=best_cycle_edges,
                    iterations=iteration,
                    cuts=len(sec_subsets),
                    sec_subsets=tuple(sec_subsets),
                    rounds=tuple(round_records),
                    exact=True,
                    solver_status=last_solver_status,
                    message=(
                        "no solution exists below the retained connected "
                        "incumbent at the declared objective granularity"
                    ),
                    wall_seconds=time.perf_counter() - total_started,
                    first_feasible_wall_seconds_upper_bound=(
                        first_feasible_wall_seconds_upper_bound
                    ),
                    best_feasible_wall_seconds_upper_bound=(
                        best_feasible_wall_seconds_upper_bound
                    ),
                )
            if best_cycle is not None:
                return SubtourMILPResult(
                    status=(
                        "feasible_solver_error"
                        if last_solver_status in {3, 4}
                        else "feasible_limit"
                    ),
                    objective=best_cycle_objective,
                    lower_bound=best_lower_bound,
                    cycle=best_cycle,
                    selected_edges=best_cycle_edges,
                    iterations=iteration,
                    cuts=len(sec_subsets),
                    sec_subsets=tuple(sec_subsets),
                    rounds=tuple(round_records),
                    exact=False,
                    solver_status=last_solver_status,
                    message=last_message,
                    wall_seconds=time.perf_counter() - total_started,
                    first_feasible_wall_seconds_upper_bound=(
                        first_feasible_wall_seconds_upper_bound
                    ),
                    best_feasible_wall_seconds_upper_bound=(
                        best_feasible_wall_seconds_upper_bound
                    ),
                )
            return SubtourMILPResult(
                status=status,
                objective=last_objective,
                lower_bound=best_lower_bound,
                cycle=None,
                selected_edges=(),
                iterations=iteration,
                cuts=len(sec_subsets),
                sec_subsets=tuple(sec_subsets),
                rounds=tuple(round_records),
                exact=False,
                solver_status=last_solver_status,
                message=last_message,
                wall_seconds=time.perf_counter() - total_started,
            )

        last_selected = tuple(
            edge for edge, value in zip(edges, result.x, strict=True) if value > 0.5
        )
        components = _components(n, last_selected)
        cycle = (
            _cycle_from_edges(n, last_selected)
            if len(components) == 1
            else None
        )
        if cycle is not None:
            cycle_objective = float(
                sum(normalized[candidate] for candidate in last_selected)
            )
            if (
                last_objective is None
                or not np.isclose(
                    last_objective,
                    cycle_objective,
                    atol=1e-6,
                    rtol=0.0,
                )
            ):
                raise RuntimeError(
                    "MILP objective disagrees with its connected incumbent"
                )
            detected_elapsed = time.perf_counter() - total_started
            if first_feasible_wall_seconds_upper_bound is None:
                first_feasible_wall_seconds_upper_bound = detected_elapsed
            if (
                best_cycle_objective is None
                or cycle_objective < best_cycle_objective
            ):
                best_cycle = cycle
                best_cycle_edges = last_selected
                best_cycle_objective = cycle_objective
                best_feasible_wall_seconds_upper_bound = detected_elapsed
            round_records.append(
                SubtourMILPRound(
                    iteration=iteration,
                    solver_status=last_solver_status,
                    objective=last_objective,
                    lower_bound=last_lower_bound,
                    component_sizes=(n,),
                    cuts_before=len(sec_subsets),
                    cuts_added=0,
                    mip_node_count=round_node_count,
                    mip_gap=round_mip_gap,
                    wall_seconds=round_wall_seconds,
                )
            )
            optimal = last_solver_status == 0
            if continue_after_feasible_limit and not optimal:
                continue
            return SubtourMILPResult(
                status="optimal" if optimal else "feasible_limit",
                objective=cycle_objective,
                lower_bound=best_lower_bound,
                cycle=cycle,
                selected_edges=last_selected,
                iterations=iteration,
                cuts=len(sec_subsets),
                sec_subsets=tuple(sec_subsets),
                rounds=tuple(round_records),
                exact=optimal,
                solver_status=last_solver_status,
                message=last_message,
                wall_seconds=time.perf_counter() - total_started,
                first_feasible_wall_seconds_upper_bound=(
                    first_feasible_wall_seconds_upper_bound
                ),
                best_feasible_wall_seconds_upper_bound=(
                    best_feasible_wall_seconds_upper_bound
                ),
            )

        # A non-optimal solve need not have found the best 2-factor for the
        # current relaxation.  Its disconnected incumbent still yields valid
        # SECs, so add them and continue while iterations remain.
        cuts_before = len(sec_subsets)
        for component in components:
            subset = _canonical_sec_subset(component, n)
            if subset is not None and subset not in seen_subsets:
                seen_subsets.add(subset)
                sec_subsets.append(subset)
        round_records.append(
            SubtourMILPRound(
                iteration=iteration,
                solver_status=last_solver_status,
                objective=last_objective,
                lower_bound=last_lower_bound,
                component_sizes=tuple(sorted(map(len, components))),
                cuts_before=cuts_before,
                cuts_added=len(sec_subsets) - cuts_before,
                mip_node_count=round_node_count,
                mip_gap=round_mip_gap,
                wall_seconds=round_wall_seconds,
            )
        )

    if best_cycle is not None:
        return SubtourMILPResult(
            status="feasible_iteration_limit",
            objective=best_cycle_objective,
            lower_bound=best_lower_bound,
            cycle=best_cycle,
            selected_edges=best_cycle_edges,
            iterations=iteration_limit,
            cuts=len(sec_subsets),
            sec_subsets=tuple(sec_subsets),
            rounds=tuple(round_records),
            exact=False,
            solver_status=last_solver_status,
            message=last_message,
            wall_seconds=time.perf_counter() - total_started,
            first_feasible_wall_seconds_upper_bound=(
                first_feasible_wall_seconds_upper_bound
            ),
            best_feasible_wall_seconds_upper_bound=(
                best_feasible_wall_seconds_upper_bound
            ),
        )
    return SubtourMILPResult(
        status="iteration_limit",
        objective=last_objective,
        lower_bound=best_lower_bound,
        cycle=None,
        selected_edges=last_selected,
        iterations=iteration_limit,
        cuts=len(sec_subsets),
        sec_subsets=tuple(sec_subsets),
        rounds=tuple(round_records),
        exact=False,
        solver_status=last_solver_status,
        message=last_message,
        wall_seconds=time.perf_counter() - total_started,
    )


__all__ = [
    "Edge",
    "SubtourMILPRound",
    "SubtourMILPResult",
    "solve_hamiltonian_cycle",
]
