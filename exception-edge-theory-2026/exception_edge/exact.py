from __future__ import annotations

from itertools import permutations
from math import inf
from operator import index
from typing import Iterable, Sequence

import numpy as np

from .geometry import tour_edges, tour_length


def held_karp_cycle(
    dist: np.ndarray, allowed_edges: set[tuple[int, int]] | None = None
) -> tuple[float, list[int]]:
    """Exact symmetric TSP by Bellman--Held--Karp dynamic programming.

    The returned tour starts at vertex 0 and does not repeat 0 at the end.
    Runtime is O(n^2 2^n); this implementation is intended for n <= 18 and
    for the paper configuration n <= 13.
    """
    n = len(dist)
    if n < 3:
        raise ValueError("TSP cycle requires at least three vertices")
    start_mask = 1
    dp: dict[tuple[int, int], float] = {(start_mask, 0): 0.0}
    parent: dict[tuple[int, int], int] = {}

    for mask in range(1 << n):
        if not (mask & start_mask):
            continue
        for last in range(n):
            state = (mask, last)
            current = dp.get(state)
            if current is None:
                continue
            for nxt in range(1, n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                if allowed_edges is not None and tuple(sorted((last, nxt))) not in allowed_edges:
                    continue
                new_mask = mask | bit
                candidate = current + float(dist[last, nxt])
                new_state = (new_mask, nxt)
                old = dp.get(new_state, inf)
                if candidate < old - 1e-15:
                    dp[new_state] = candidate
                    parent[new_state] = last

    full = (1 << n) - 1
    best_cost = inf
    best_last = -1
    for last in range(1, n):
        if allowed_edges is not None and (0, last) not in allowed_edges:
            continue
        if (full, last) not in dp:
            continue
        candidate = dp[(full, last)] + float(dist[last, 0])
        if candidate < best_cost - 1e-15:
            best_cost = candidate
            best_last = last

    if best_last < 0:
        return inf, []

    reverse_path: list[int] = []
    mask = full
    last = best_last
    while last != 0:
        reverse_path.append(last)
        previous = parent[(mask, last)]
        mask ^= 1 << last
        last = previous
    tour = [0] + list(reversed(reverse_path))
    return float(best_cost), tour


def held_karp_cycle_by_crossings(
    dist: np.ndarray, side_a: Sequence[int]
) -> dict[int, float]:
    """Exact TSP value for every attainable cut-crossing count."""
    n = len(dist)
    side_a = set(map(int, side_a))
    if not side_a or side_a == set(range(n)):
        raise ValueError("side_a must define a nontrivial cut")

    def crosses(u: int, v: int) -> int:
        return int((u in side_a) != (v in side_a))

    layers: list[dict[tuple[int, int], float]] = [
        {} for _ in range(1 << n)
    ]
    layers[1][(0, 0)] = 0.0
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        for (last, count), current in list(layers[mask].items()):
            for nxt in range(1, n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_state = (nxt, count + crosses(last, nxt))
                candidate = current + float(dist[last, nxt])
                if candidate < layers[new_mask].get(new_state, inf):
                    layers[new_mask][new_state] = candidate

    full = (1 << n) - 1
    result: dict[int, float] = {}
    for (last, path_count), value in layers[full].items():
        if last == 0:
            continue
        total_count = path_count + crosses(last, 0)
        candidate = value + float(dist[last, 0])
        result[total_count] = min(result.get(total_count, inf), candidate)
    return dict(sorted(result.items()))


def held_karp_cycle_by_exception_count(
    dist: np.ndarray,
    baseline_edges: Iterable[tuple[int, int]],
) -> dict[int, tuple[float, list[int]]]:
    """Return the exact TSP spectrum indexed by baseline-exception count.

    For every attainable ``q``, the returned mapping contains
    ``q: (value, tour)``, where ``value`` is the minimum Hamiltonian-cycle
    cost among cycles using exactly ``q`` edges outside ``baseline_edges``.
    Each witness ``tour`` starts at vertex 0 and does not repeat 0.

    Baseline edges are undirected: endpoints are normalized to increasing
    order, and duplicate or reversed copies are harmless.  Self-loops,
    non-integral endpoints, and endpoints outside ``range(len(dist))`` are
    rejected.  The distance matrix must be finite, nonnegative, square, and
    exactly symmetric.

    Equal-cost dynamic-programming states retain the lexicographically
    smallest path.  Final cycles are also canonicalized against reversal, so
    witness selection is deterministic and independent of baseline-edge
    iteration order.  The arithmetic DP has ``O(n^3 2^n)`` transitions and
    ``O(n^2 2^n)`` scalar states.  This implementation stores and copies an
    ``O(n)`` path tuple in every state for deterministic witnesses, so its
    conservative Python-level bounds are ``O(n^4 2^n)`` time and
    ``O(n^3 2^n)`` space.  Floating-point costs remain subject to ordinary
    floating-point arithmetic.
    """
    matrix = np.asarray(dist, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dist must be a square matrix")
    n = int(matrix.shape[0])
    if n < 3:
        raise ValueError("TSP cycle requires at least three vertices")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("dist must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError("dist must be nonnegative")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("dist must be exactly symmetric")

    baseline: set[tuple[int, int]] = set()
    try:
        edge_iterator = iter(baseline_edges)
    except TypeError as exc:
        raise ValueError("baseline_edges must be an iterable of edge pairs") from exc
    for candidate in edge_iterator:
        try:
            raw_u, raw_v = candidate
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each baseline edge must contain exactly two endpoints"
            ) from exc
        try:
            u = index(raw_u)
            v = index(raw_v)
        except TypeError as exc:
            raise ValueError("baseline edge endpoints must be integers") from exc
        if u == v:
            raise ValueError("baseline edges cannot contain self-loops")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("baseline edge endpoint is outside the distance matrix")
        baseline.add((u, v) if u < v else (v, u))

    # One layer per visited-vertex mask.  A state stores the best cost and
    # lexicographically smallest path for a fixed (last vertex, exception
    # count).  Keeping the path in the state makes tie handling explicit.
    layers: list[
        dict[tuple[int, int], tuple[float, tuple[int, ...]]]
    ] = [{} for _ in range(1 << n)]
    layers[1][(0, 0)] = (0.0, (0,))

    for mask in range(1 << n):
        if not (mask & 1):
            continue
        for (last, count), (current, path) in sorted(
            layers[mask].items()
        ):
            for nxt in range(1, n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                candidate_edge = (
                    (last, nxt) if last < nxt else (nxt, last)
                )
                new_count = count + int(candidate_edge not in baseline)
                new_mask = mask | bit
                new_state = (nxt, new_count)
                candidate_value = current + float(matrix[last, nxt])
                candidate_path = path + (nxt,)
                previous = layers[new_mask].get(new_state)
                if (
                    previous is None
                    or candidate_value < previous[0]
                    or (
                        candidate_value == previous[0]
                        and candidate_path < previous[1]
                    )
                ):
                    layers[new_mask][new_state] = (
                        candidate_value,
                        candidate_path,
                    )

    full = (1 << n) - 1
    result: dict[int, tuple[float, tuple[int, ...]]] = {}
    for (last, path_count), (path_value, path) in sorted(
        layers[full].items()
    ):
        closing_edge = (0, last)
        total_count = path_count + int(closing_edge not in baseline)
        reverse = (0, *reversed(path[1:]))
        canonical = min(path, reverse)
        candidate_value = path_value + float(matrix[last, 0])
        previous = result.get(total_count)
        if (
            previous is None
            or candidate_value < previous[0]
            or (
                candidate_value == previous[0]
                and canonical < previous[1]
            )
        ):
            result[total_count] = (candidate_value, canonical)

    return {
        count: (float(value), list(path))
        for count, (value, path) in sorted(result.items())
    }


def reference_edge_forbid_oracle(
    dist: np.ndarray,
    optimum: float,
    reference_tour: Sequence[int],
    tolerance: float = 1e-9,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float], float, bool]:
    """Identify edges present in every optimum by leave-one-edge-out solves.

    A non-reference edge cannot belong to every optimum because the supplied
    reference optimum omits it.  Therefore only the reference-tour edges need
    a forbidden-edge solve.  If every reference edge is forced, the optimum
    cycle is unique up to orientation.
    """
    complete = set(
        (u, v) for u in range(len(dist)) for v in range(u + 1, len(dist))
    )
    reference = tour_edges(reference_tour)
    tol_abs = tolerance * max(1.0, abs(float(optimum)))
    forbidden_values: dict[tuple[int, int], float] = {}
    forced: set[tuple[int, int]] = set()
    increases: list[float] = []
    for candidate in sorted(reference):
        value, _ = held_karp_cycle(
            dist,
            allowed_edges=complete - {candidate},
        )
        forbidden_values[candidate] = value
        increase = float(value - optimum)
        increases.append(increase)
        if value > optimum + tol_abs:
            forced.add(candidate)
    uniqueness_margin = min(increases, default=inf)
    unique = forced == reference
    return forced, forbidden_values, uniqueness_margin, unique


def all_fixed_endpoint_path_costs(
    dist: np.ndarray, vertices: Sequence[int]
) -> dict[tuple[int, int], float]:
    """All exact Hamiltonian-path costs with distinct prescribed endpoints."""
    vertices = tuple(map(int, vertices))
    m = len(vertices)
    if m < 2:
        return {}
    result: dict[tuple[int, int], float] = {}
    for local_start, global_start in enumerate(vertices):
        dp: dict[tuple[int, int], float] = {
            (1 << local_start, local_start): 0.0
        }
        for mask in range(1 << m):
            if not (mask & (1 << local_start)):
                continue
            for last in range(m):
                current = dp.get((mask, last))
                if current is None:
                    continue
                for nxt in range(m):
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    new_state = (mask | bit, nxt)
                    candidate = current + float(dist[vertices[last], vertices[nxt]])
                    if candidate < dp.get(new_state, inf):
                        dp[new_state] = candidate
        full = (1 << m) - 1
        for local_end, global_end in enumerate(vertices):
            if local_end == local_start:
                continue
            key = tuple(sorted((global_start, global_end)))
            result[key] = min(result.get(key, inf), dp[(full, local_end)])
    return result


def free_path_costs_by_mask(
    dist: np.ndarray, vertices: Sequence[int]
) -> list[float]:
    """Minimum Hamiltonian-path cost for every nonempty induced subset."""
    vertices = tuple(map(int, vertices))
    m = len(vertices)
    size = 1 << m
    free = [inf] * size
    free[0] = 0.0
    for start in range(m):
        dp: dict[tuple[int, int], float] = {(1 << start, start): 0.0}
        free[1 << start] = 0.0
        for mask in range(size):
            if not (mask & (1 << start)):
                continue
            for last in range(m):
                current = dp.get((mask, last))
                if current is None:
                    continue
                if current < free[mask]:
                    free[mask] = current
                for nxt in range(m):
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    new_state = (mask | bit, nxt)
                    candidate = current + float(dist[vertices[last], vertices[nxt]])
                    if candidate < dp.get(new_state, inf):
                        dp[new_state] = candidate
    return free


def path_cover_costs(
    dist: np.ndarray, vertices: Sequence[int]
) -> dict[int, float]:
    """Exact minimum cost of q disjoint paths covering the vertex set.

    Isolated vertices count as paths.  A canonical-subset recurrence avoids
    counting different orders of the same partition.
    """
    vertices = tuple(map(int, vertices))
    m = len(vertices)
    if m == 0:
        return {0: 0.0}
    free = free_path_costs_by_mask(dist, vertices)
    full = (1 << m) - 1
    previous = [inf] * (1 << m)
    previous[0] = 0.0
    answer: dict[int, float] = {}

    for q in range(1, m + 1):
        current = [inf] * (1 << m)
        for mask in range(1, 1 << m):
            if mask.bit_count() < q:
                continue
            anchor = mask & -mask
            sub = mask
            while sub:
                if sub & anchor:
                    remainder = mask ^ sub
                    if previous[remainder] < inf and free[sub] < inf:
                        candidate = previous[remainder] + free[sub]
                        if candidate < current[mask]:
                            current[mask] = candidate
                sub = (sub - 1) & mask
        answer[q] = current[full]
        previous = current
    return answer


def enumerate_optimal_tours(
    dist: np.ndarray, tolerance: float = 1e-9, max_n: int = 10
) -> tuple[float, list[list[int]]]:
    """Enumerate canonical optimal tours; used only for tie-sensitive tests."""
    n = len(dist)
    if n > max_n:
        raise ValueError(f"enumeration is restricted to n <= {max_n}")
    best = inf
    optima: list[list[int]] = []
    for tail in permutations(range(1, n)):
        if tail[0] > tail[-1]:
            continue  # reverse cycles are identical
        tour = [0, *tail]
        value = tour_length(tour, dist)
        if value < best - tolerance:
            best = value
            optima = [tour]
        elif abs(value - best) <= tolerance:
            optima.append(tour)
    return float(best), optima


def optimal_edge_sets(tours: Sequence[Sequence[int]]) -> tuple[set, set]:
    if not tours:
        return set(), set()
    sets = [tour_edges(tour) for tour in tours]
    union = set().union(*sets)
    intersection = set(sets[0]).intersection(*sets[1:])
    return union, intersection
