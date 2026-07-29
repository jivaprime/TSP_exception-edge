"""Deterministic local-basin interventions for the LIN318 experiment.

This module deliberately has no dependency on the LIN318 reference tour or on
the published optimum.  It reconstructs the 42,210 strict-1,500 incumbent from
the sealed Colab output, performs candidate-restricted 2/3-opt, and tests a
target edge by an atomic insertion followed by locked and unlocked relaxation.

The primary intervention locks every edge added by the atomic move.  This
prevents the first local-search step from simply undoing the perturbation and
keeps attribution at the indivisible move-unit level.  Whether one particular
edge deserves individual credit is a separate, post-intervention question.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import zipfile


Edge = tuple[int, int]

TARGET_CANDIDATE_MEMBER = (
    "benchmark_pilot/runs/static_local_b512__r01/solver_output/"
    "candidate_edges/round_00.csv"
)
SOURCE_FACTOR_MEMBER = (
    "benchmark_pilot/runs/static_shortest_b512__r01/solver_output/"
    "milp_audit/round_00.json"
)
TARGET_CANDIDATE_SHA256 = (
    "aa75561a53100148eb69b45fef04bd690e5b0f7a0db9d8a573b85ecb1d3d4a69"
)


def edge(u: int, v: int) -> Edge:
    """Return a canonical undirected edge."""
    a = int(u)
    b = int(v)
    if a == b:
        raise ValueError("self loops are not valid TSP edges")
    return (a, b) if a < b else (b, a)


def _normalize_costs(edge_costs: Mapping[Edge, int]) -> dict[Edge, int]:
    normalized: dict[Edge, int] = {}
    for pair, value in edge_costs.items():
        if len(pair) != 2:
            raise ValueError(f"invalid edge key: {pair!r}")
        candidate = edge(pair[0], pair[1])
        cost = int(value)
        if cost < 0:
            raise ValueError(f"negative edge cost for {candidate}")
        previous = normalized.get(candidate)
        if previous is not None and previous != cost:
            raise ValueError(f"conflicting costs for {candidate}")
        normalized[candidate] = cost
    if not normalized:
        raise ValueError("edge_costs must not be empty")
    return normalized


def _vertex_count(edge_costs: Mapping[Edge, int]) -> int:
    vertices = {vertex for pair in edge_costs for vertex in pair}
    if not vertices:
        raise ValueError("candidate graph has no vertices")
    n = max(vertices) + 1
    if vertices != set(range(n)):
        raise ValueError("candidate vertices must be exactly range(n)")
    return n


def tour_edge_set(tour: Sequence[int]) -> frozenset[Edge]:
    normalized = tuple(int(vertex) for vertex in tour)
    if len(normalized) < 3:
        raise ValueError("a Hamiltonian cycle needs at least three vertices")
    return frozenset(
        edge(normalized[index], normalized[(index + 1) % len(normalized)])
        for index in range(len(normalized))
    )


def canonical_tour(tour: Sequence[int]) -> tuple[int, ...]:
    """Canonicalize a cycle up to rotation and reversal."""
    normalized = tuple(int(vertex) for vertex in tour)
    if not normalized:
        raise ValueError("tour must not be empty")
    start = min(normalized)
    position = normalized.index(start)
    forward = normalized[position:] + normalized[:position]
    reversed_cycle = tuple(reversed(normalized))
    reverse_position = reversed_cycle.index(start)
    backward = (
        reversed_cycle[reverse_position:] + reversed_cycle[:reverse_position]
    )
    return min(forward, backward)


def validate_tour(
    tour: Sequence[int],
    edge_costs: Mapping[Edge, int],
    *,
    locked_edges: Iterable[Edge] = (),
) -> dict[str, object]:
    """Check Hamiltonicity, candidate membership, and lock preservation."""
    costs = _normalize_costs(edge_costs)
    n = _vertex_count(costs)
    normalized = tuple(int(vertex) for vertex in tour)
    if len(normalized) != n or set(normalized) != set(range(n)):
        raise ValueError("tour is not a permutation of range(n)")
    used = tour_edge_set(normalized)
    if len(used) != n:
        raise ValueError("tour does not contain n distinct undirected edges")
    missing = sorted(used - set(costs))
    if missing:
        raise ValueError(f"tour uses edges outside the candidate graph: {missing}")
    locks = frozenset(edge(*pair) for pair in locked_edges)
    lost = sorted(locks - used)
    if lost:
        raise ValueError(f"tour lost locked edges: {lost}")
    value = sum(costs[pair] for pair in used)
    return {
        "hamiltonian": True,
        "candidate_membership": True,
        "locked_edges_preserved": True,
        "vertex_count": n,
        "edge_count": len(used),
        "cost": int(value),
    }


def tour_cost(
    tour: Sequence[int],
    edge_costs: Mapping[Edge, int],
) -> int:
    return int(validate_tour(tour, edge_costs)["cost"])


@dataclass(frozen=True)
class AtomicMove:
    """One valid 2-opt or genuine 3-opt reconnection."""

    kind: str
    pattern: str
    cut_indices: tuple[int, ...]
    removed_edges: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]
    delta: int
    resulting_tour: tuple[int, ...]

    def result_dict(self, *, include_tour: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": self.kind,
            "pattern": self.pattern,
            "cut_indices": self.cut_indices,
            "removed_edges": self.removed_edges,
            "added_edges": self.added_edges,
            "delta": self.delta,
        }
        if include_tour:
            value["resulting_tour"] = self.resulting_tour
        return value


def _move_key(move: AtomicMove) -> tuple[object, ...]:
    kind_rank = 0 if move.kind == "2opt" else 1
    return (
        move.delta,
        kind_rank,
        move.added_edges,
        move.removed_edges,
        move.pattern,
        move.cut_indices,
    )


def _candidate_neighbors(
    edge_costs: Mapping[Edge, int],
    n: int,
) -> tuple[tuple[int, ...], ...]:
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for u, v in edge_costs:
        neighbors[u].add(v)
        neighbors[v].add(u)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _two_opt_move(
    tour: tuple[int, ...],
    i: int,
    j: int,
    costs: Mapping[Edge, int],
) -> AtomicMove | None:
    n = len(tour)
    if j <= i + 1 or (i == 0 and j == n - 1):
        return None
    a = tour[i]
    b = tour[(i + 1) % n]
    c = tour[j]
    d = tour[(j + 1) % n]
    removed = tuple(sorted((edge(a, b), edge(c, d))))
    added = tuple(sorted((edge(a, c), edge(b, d))))
    if any(pair not in costs for pair in added):
        return None
    delta = sum(costs[pair] for pair in added) - sum(
        costs[pair] for pair in removed
    )
    result = tour[: i + 1] + tuple(reversed(tour[i + 1 : j + 1])) + tour[j + 1 :]
    return AtomicMove(
        kind="2opt",
        pattern="reverse_middle",
        cut_indices=(i, j),
        removed_edges=removed,
        added_edges=added,
        delta=int(delta),
        resulting_tour=result,
    )


_THREE_OPT_PATTERN_RANK = {
    "reverse_x_reverse_y": 0,
    "y_then_x": 1,
    "reverse_y_then_x": 2,
    "y_then_reverse_x": 3,
}


def _three_opt_move(
    tour: tuple[int, ...],
    pattern: str,
    i: int,
    j: int,
    k: int,
    costs: Mapping[Edge, int],
) -> AtomicMove | None:
    n = len(tour)
    if (
        i < 0
        or j <= i + 1
        or k <= j + 1
        or k >= n
        or (i == 0 and k == n - 1)
    ):
        return None
    a = tour[i]
    b = tour[i + 1]
    c = tour[j]
    d = tour[j + 1]
    e = tour[k]
    f = tour[(k + 1) % n]
    removed = tuple(sorted((edge(a, b), edge(c, d), edge(e, f))))
    prefix = tour[: i + 1]
    x = tour[i + 1 : j + 1]
    y = tour[j + 1 : k + 1]
    suffix = tour[k + 1 :]
    if pattern == "reverse_x_reverse_y":
        added = tuple(sorted((edge(a, c), edge(b, e), edge(d, f))))
        result = prefix + tuple(reversed(x)) + tuple(reversed(y)) + suffix
    elif pattern == "y_then_x":
        added = tuple(sorted((edge(a, d), edge(e, b), edge(c, f))))
        result = prefix + y + x + suffix
    elif pattern == "reverse_y_then_x":
        added = tuple(sorted((edge(a, e), edge(d, b), edge(c, f))))
        result = prefix + tuple(reversed(y)) + x + suffix
    elif pattern == "y_then_reverse_x":
        added = tuple(sorted((edge(a, d), edge(e, c), edge(b, f))))
        result = prefix + y + tuple(reversed(x)) + suffix
    else:
        raise ValueError(f"unknown genuine 3-opt pattern: {pattern!r}")
    if any(pair not in costs for pair in added):
        return None
    delta = sum(costs[pair] for pair in added) - sum(
        costs[pair] for pair in removed
    )
    return AtomicMove(
        kind="3opt",
        pattern=pattern,
        cut_indices=(i, j, k),
        removed_edges=removed,
        added_edges=added,
        delta=int(delta),
        resulting_tour=result,
    )


def _best_two_opt_move(
    tour: tuple[int, ...],
    costs: Mapping[Edge, int],
    neighbors: tuple[tuple[int, ...], ...],
    *,
    locked_edges: frozenset[Edge],
    require_added_edge: Edge | None = None,
    improving_only: bool,
) -> AtomicMove | None:
    n = len(tour)
    position = {vertex: index for index, vertex in enumerate(tour)}
    best: AtomicMove | None = None
    for i, a in enumerate(tour):
        for c in neighbors[a]:
            j = position[c]
            move = _two_opt_move(tour, i, j, costs)
            if move is None:
                continue
            if locked_edges.intersection(move.removed_edges):
                continue
            if (
                require_added_edge is not None
                and require_added_edge not in move.added_edges
            ):
                continue
            if improving_only and move.delta >= 0:
                continue
            if best is None or _move_key(move) < _move_key(best):
                best = move
    return best


def _consider_three_opt(
    best: AtomicMove | None,
    tour: tuple[int, ...],
    pattern: str,
    i: int,
    j: int,
    k: int,
    costs: Mapping[Edge, int],
    *,
    locked_edges: frozenset[Edge],
    require_added_edge: Edge | None,
    improving_only: bool,
) -> AtomicMove | None:
    move = _three_opt_move(tour, pattern, i, j, k, costs)
    if move is None:
        return best
    if locked_edges.intersection(move.removed_edges):
        return best
    if require_added_edge is not None and require_added_edge not in move.added_edges:
        return best
    if improving_only and move.delta >= 0:
        return best
    if best is None or _move_key(move) < _move_key(best):
        return move
    return best


def _best_three_opt_move(
    tour: tuple[int, ...],
    costs: Mapping[Edge, int],
    neighbors: tuple[tuple[int, ...], ...],
    *,
    locked_edges: frozenset[Edge],
    require_added_edge: Edge | None = None,
    improving_only: bool,
) -> AtomicMove | None:
    """Find the best of the four 2-opt-irreducible 3-opt patterns."""
    n = len(tour)
    position = {vertex: index for index, vertex in enumerate(tour)}
    best: AtomicMove | None = None
    for i, a in enumerate(tour):
        b = tour[(i + 1) % n]

        # reverse_x_reverse_y: A-C, B-E, D-F
        for c in neighbors[a]:
            j = position[c]
            if j <= i + 1 or j >= n - 1:
                continue
            d = tour[j + 1]
            for e in neighbors[b]:
                k = position[e]
                if k <= j + 1 or (i == 0 and k == n - 1):
                    continue
                f = tour[(k + 1) % n]
                if edge(d, f) not in costs:
                    continue
                best = _consider_three_opt(
                    best,
                    tour,
                    "reverse_x_reverse_y",
                    i,
                    j,
                    k,
                    costs,
                    locked_edges=locked_edges,
                    require_added_edge=require_added_edge,
                    improving_only=improving_only,
                )

        # y_then_x: A-D, E-B, C-F
        for d in neighbors[a]:
            d_position = position[d]
            if d_position == 0:
                continue
            j = d_position - 1
            if j <= i + 1 or j >= n - 1:
                continue
            c = tour[j]
            for e in neighbors[b]:
                k = position[e]
                if k <= j + 1 or (i == 0 and k == n - 1):
                    continue
                f = tour[(k + 1) % n]
                if edge(c, f) not in costs:
                    continue
                best = _consider_three_opt(
                    best,
                    tour,
                    "y_then_x",
                    i,
                    j,
                    k,
                    costs,
                    locked_edges=locked_edges,
                    require_added_edge=require_added_edge,
                    improving_only=improving_only,
                )

        # reverse_y_then_x: A-E, D-B, C-F
        for e in neighbors[a]:
            k = position[e]
            if k <= i + 3:
                continue
            for d in neighbors[b]:
                d_position = position[d]
                if d_position == 0:
                    continue
                j = d_position - 1
                if (
                    j <= i + 1
                    or k <= j + 1
                    or (i == 0 and k == n - 1)
                ):
                    continue
                c = tour[j]
                f = tour[(k + 1) % n]
                if edge(c, f) not in costs:
                    continue
                best = _consider_three_opt(
                    best,
                    tour,
                    "reverse_y_then_x",
                    i,
                    j,
                    k,
                    costs,
                    locked_edges=locked_edges,
                    require_added_edge=require_added_edge,
                    improving_only=improving_only,
                )

        # y_then_reverse_x: A-D, E-C, B-F
        for d in neighbors[a]:
            d_position = position[d]
            if d_position == 0:
                continue
            j = d_position - 1
            if j <= i + 1 or j >= n - 1:
                continue
            c = tour[j]
            for e in neighbors[c]:
                k = position[e]
                if k <= j + 1 or (i == 0 and k == n - 1):
                    continue
                f = tour[(k + 1) % n]
                if edge(b, f) not in costs:
                    continue
                best = _consider_three_opt(
                    best,
                    tour,
                    "y_then_reverse_x",
                    i,
                    j,
                    k,
                    costs,
                    locked_edges=locked_edges,
                    require_added_edge=require_added_edge,
                    improving_only=improving_only,
                )
    return best


def strict_two_three_opt(
    tour: Sequence[int],
    edge_costs: Mapping[Edge, int],
    *,
    locked_edges: Iterable[Edge] = (),
) -> dict[str, object]:
    """Converge under candidate-restricted best-improvement 2/3-opt.

    The four 3-opt patterns are precisely the genuine patterns not reducible to
    one 2-opt move.  The remaining nontrivial 3-opt reconnections are covered
    by the preceding exhaustive 2-opt convergence.
    """
    costs = _normalize_costs(edge_costs)
    n = _vertex_count(costs)
    current = canonical_tour(tour)
    locks = frozenset(edge(*pair) for pair in locked_edges)
    initial_validation = validate_tour(current, costs, locked_edges=locks)
    if len(current) != n:
        raise ValueError("tour and candidate graph dimensions disagree")
    neighbors = _candidate_neighbors(costs, n)
    initial_cost = int(initial_validation["cost"])
    current_cost = initial_cost
    two_opt_moves = 0
    three_opt_moves = 0

    while True:
        while True:
            move = _best_two_opt_move(
                current,
                costs,
                neighbors,
                locked_edges=locks,
                improving_only=True,
            )
            if move is None:
                break
            before = current_cost
            current = move.resulting_tour
            current_cost = tour_cost(current, costs)
            if current_cost != before + move.delta or current_cost >= before:
                raise RuntimeError("2-opt delta or strict decrease invariant failed")
            validate_tour(current, costs, locked_edges=locks)
            two_opt_moves += 1

        move = _best_three_opt_move(
            current,
            costs,
            neighbors,
            locked_edges=locks,
            improving_only=True,
        )
        if move is None:
            break
        before = current_cost
        current = move.resulting_tour
        current_cost = tour_cost(current, costs)
        if current_cost != before + move.delta or current_cost >= before:
            raise RuntimeError("3-opt delta or strict decrease invariant failed")
        validate_tour(current, costs, locked_edges=locks)
        three_opt_moves += 1

    current = canonical_tour(current)
    final_validation = validate_tour(current, costs, locked_edges=locks)
    if int(final_validation["cost"]) != current_cost:
        raise RuntimeError("canonicalization changed tour cost")
    return {
        "schema": "strict-candidate-two-three-opt-v1",
        "tour": current,
        "initial_cost": initial_cost,
        "final_cost": current_cost,
        "two_opt_moves": two_opt_moves,
        "three_opt_moves": three_opt_moves,
        "locked_edges": tuple(sorted(locks)),
        "strictly_nonincreasing": current_cost <= initial_cost,
        "locally_optimal_2opt": True,
        "locally_optimal_genuine_3opt": True,
        "validation": final_validation,
    }


def minimum_barrier_insertion(
    tour: Sequence[int],
    target_edge: Edge,
    edge_costs: Mapping[Edge, int],
    *,
    move_kinds: Iterable[str] = ("2opt", "3opt"),
    locked_edges: Iterable[Edge] = (),
) -> AtomicMove:
    """Return the minimum-delta atomic 2/3-opt move that inserts ``target``.

    ``locked_edges`` makes sequential, multi-edge interventions auditable:
    an insertion may not remove a target or companion edge admitted by an
    earlier atomic unit.
    """
    costs = _normalize_costs(edge_costs)
    n = _vertex_count(costs)
    current = canonical_tour(tour)
    locks = frozenset(edge(*pair) for pair in locked_edges)
    validate_tour(current, costs, locked_edges=locks)
    target = edge(*target_edge)
    if target not in costs:
        raise ValueError("target edge is outside the candidate graph")
    if target in tour_edge_set(current):
        raise ValueError("target edge is already present in the tour")
    requested = frozenset(str(kind) for kind in move_kinds)
    if not requested or not requested.issubset({"2opt", "3opt"}):
        raise ValueError("move_kinds must be a nonempty subset of {'2opt','3opt'}")
    neighbors = _candidate_neighbors(costs, n)
    candidates: list[AtomicMove] = []
    if "2opt" in requested:
        move = _best_two_opt_move(
            current,
            costs,
            neighbors,
            locked_edges=locks,
            require_added_edge=target,
            improving_only=False,
        )
        if move is not None:
            candidates.append(move)
    if "3opt" in requested:
        move = _best_three_opt_move(
            current,
            costs,
            neighbors,
            locked_edges=locks,
            require_added_edge=target,
            improving_only=False,
        )
        if move is not None:
            candidates.append(move)
    if not candidates:
        raise ValueError(
            f"no requested atomic insertion of target edge {target} exists"
        )
    chosen = min(candidates, key=_move_key)
    before = tour_cost(current, costs)
    after_validation = validate_tour(chosen.resulting_tour, costs)
    after = int(after_validation["cost"])
    if after != before + chosen.delta:
        raise RuntimeError("atomic insertion delta invariant failed")
    if target not in tour_edge_set(chosen.resulting_tour):
        raise RuntimeError("atomic insertion did not add its target edge")
    return chosen


def forced_edge_bundle_relaxation(
    tour: Sequence[int],
    target_edges: Iterable[Edge],
    edge_costs: Mapping[Edge, int],
    *,
    move_kinds: Iterable[str] = ("2opt", "3opt"),
    lock_policy: str = "added_set",
) -> dict[str, object]:
    """Insert a predeclared edge bundle, descend with locks, then release.

    This is the multi-edge extension of :func:`forced_edge_relaxation`.
    Targets are processed in the caller's declared order.  Under the primary
    ``added_set`` policy, every edge added by every atomic unit remains locked
    until all requested targets have been inserted and the constrained
    2/3-opt descent has converged.
    """
    if lock_policy not in {"added_set", "target_only"}:
        raise ValueError(
            "lock_policy must be either 'added_set' or 'target_only'"
        )
    costs = _normalize_costs(edge_costs)
    initial = canonical_tour(tour)
    initial_cost = tour_cost(initial, costs)
    ordered_targets = tuple(dict.fromkeys(edge(*pair) for pair in target_edges))
    if not ordered_targets:
        raise ValueError("target_edges must not be empty")
    missing_candidates = [target for target in ordered_targets if target not in costs]
    if missing_candidates:
        raise ValueError(
            f"bundle targets outside the candidate graph: {missing_candidates}"
        )

    current = initial
    locks: set[Edge] = set()
    moves: list[dict[str, object]] = []
    already_present: list[Edge] = []
    for target in ordered_targets:
        if target in tour_edge_set(current):
            locks.add(target)
            already_present.append(target)
            continue
        move = minimum_barrier_insertion(
            current,
            target,
            costs,
            move_kinds=move_kinds,
            locked_edges=locks,
        )
        current = move.resulting_tour
        if lock_policy == "added_set":
            locks.update(move.added_edges)
        else:
            locks.add(target)
        validate_tour(current, costs, locked_edges=locks)
        moves.append(move.result_dict())

    kicked_cost = tour_cost(current, costs)
    locked = strict_two_three_opt(current, costs, locked_edges=locks)
    locked_tour = tuple(int(vertex) for vertex in locked["tour"])
    locked_cost = int(locked["final_cost"])
    released = strict_two_three_opt(locked_tour, costs)
    final_tour = tuple(int(vertex) for vertex in released["tour"])
    final_cost = int(released["final_cost"])
    final_edges = tour_edge_set(final_tour)
    return {
        "schema": "forced-edge-bundle-basin-relaxation-v1",
        "target_edges": ordered_targets,
        "already_present_targets": tuple(already_present),
        "atomic_moves": moves,
        "lock_policy": lock_policy,
        "locked_edges": tuple(sorted(locks)),
        "initial_tour": initial,
        "initial_cost": initial_cost,
        "kicked_tour": current,
        "kicked_cost": kicked_cost,
        "locked_relaxation": locked,
        "locked_cost": locked_cost,
        "unlocked_relaxation": released,
        "final_tour": final_tour,
        "final_cost": final_cost,
        "escape_gain": initial_cost - final_cost,
        "escaped_to_better_basin": final_cost < initial_cost,
        "surviving_targets": tuple(
            target for target in ordered_targets if target in final_edges
        ),
        "all_targets_survive_final_relaxation": all(
            target in final_edges for target in ordered_targets
        ),
        "invariants": {
            "all_targets_present_before_locked_descent": all(
                target in tour_edge_set(current) for target in ordered_targets
            ),
            "all_locks_preserved_during_locked_descent": set(locks).issubset(
                tour_edge_set(locked_tour)
            ),
            "locked_descent_nonincreasing": locked_cost <= kicked_cost,
            "unlocked_descent_nonincreasing": final_cost <= locked_cost,
            "final_hamiltonian": bool(
                validate_tour(final_tour, costs)["hamiltonian"]
            ),
            "final_candidate_membership": bool(
                validate_tour(final_tour, costs)["candidate_membership"]
            ),
        },
    }


def forced_edge_relaxation(
    tour: Sequence[int],
    target_edge: Edge,
    edge_costs: Mapping[Edge, int],
    *,
    move_kinds: Iterable[str] = ("2opt", "3opt"),
    lock_policy: str = "added_set",
) -> dict[str, object]:
    """Insert a target, relax with locks, then release and relax again."""
    if lock_policy != "added_set":
        raise ValueError("the audited primary protocol requires lock_policy='added_set'")
    costs = _normalize_costs(edge_costs)
    initial = canonical_tour(tour)
    initial_validation = validate_tour(initial, costs)
    initial_cost = int(initial_validation["cost"])
    target = edge(*target_edge)
    move = minimum_barrier_insertion(
        initial,
        target,
        costs,
        move_kinds=move_kinds,
    )
    kicked = move.resulting_tour
    kicked_validation = validate_tour(
        kicked,
        costs,
        locked_edges=move.added_edges,
    )
    kicked_cost = int(kicked_validation["cost"])
    if kicked_cost != initial_cost + move.delta:
        raise RuntimeError("kick cost invariant failed")

    locked = strict_two_three_opt(
        kicked,
        costs,
        locked_edges=move.added_edges,
    )
    locked_cost = int(locked["final_cost"])
    if locked_cost > kicked_cost:
        raise RuntimeError("locked relaxation increased the objective")
    locked_tour = tuple(int(vertex) for vertex in locked["tour"])
    locked_edge_set = tour_edge_set(locked_tour)
    if not set(move.added_edges).issubset(locked_edge_set):
        raise RuntimeError("locked relaxation lost an added edge")

    released = strict_two_three_opt(locked_tour, costs)
    final_cost = int(released["final_cost"])
    if final_cost > locked_cost:
        raise RuntimeError("unlocked relaxation increased the objective")
    final_tour = tuple(int(vertex) for vertex in released["tour"])
    final_edges = tour_edge_set(final_tour)

    return {
        "schema": "forced-edge-basin-relaxation-v1",
        "target_edge": target,
        "candidate_edge_count": len(costs),
        "initial_tour": initial,
        "initial_cost": initial_cost,
        "atomic_move": move.result_dict(),
        "immediate_barrier": move.delta,
        "kicked_cost": kicked_cost,
        "lock_policy": lock_policy,
        "locked_relaxation": locked,
        "locked_cost": locked_cost,
        "unlocked_relaxation": released,
        "final_tour": final_tour,
        "final_cost": final_cost,
        "escape_gain": initial_cost - final_cost,
        "escaped_to_better_basin": final_cost < initial_cost,
        "target_survives_locked_relaxation": target in locked_edge_set,
        "target_survives_final_relaxation": target in final_edges,
        "invariants": {
            "target_inserted": target in tour_edge_set(kicked),
            "added_set_preserved_while_locked": set(
                move.added_edges
            ).issubset(locked_edge_set),
            "kick_delta_exact": kicked_cost == initial_cost + move.delta,
            "locked_descent_nonincreasing": locked_cost <= kicked_cost,
            "unlocked_descent_nonincreasing": final_cost <= locked_cost,
            "final_hamiltonian": bool(
                validate_tour(final_tour, costs)["hamiltonian"]
            ),
            "final_candidate_membership": bool(
                validate_tour(final_tour, costs)["candidate_membership"]
            ),
        },
    }


def _factor_components(
    factor_edges: Iterable[Edge],
    n: int,
) -> tuple[tuple[int, ...], ...]:
    adjacency: list[list[int]] = [[] for _ in range(n)]
    normalized = sorted(set(edge(*pair) for pair in factor_edges))
    for u, v in normalized:
        adjacency[u].append(v)
        adjacency[v].append(u)
    if any(len(values) != 2 for values in adjacency):
        raise ValueError("selected factor is not 2-regular")
    for values in adjacency:
        values.sort()
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(n):
        if start in seen:
            continue
        cycle: list[int] = []
        previous = -1
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            left, right = adjacency[current]
            following = left if left != previous else right
            previous, current = current, following
        if current != start:
            raise ValueError("selected factor contains a noncycle component")
        components.append(tuple(cycle))
    return tuple(components)


def _patch_factor(
    factor_edges: Iterable[Edge],
    edge_costs: Mapping[Edge, int],
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...]]:
    """Merge cycles using the deterministic cheapest admissible 2-edge patch."""
    costs = _normalize_costs(edge_costs)
    n = _vertex_count(costs)
    current = set(edge(*pair) for pair in factor_edges)
    if len(current) != n:
        raise ValueError("a spanning 2-factor must contain exactly n edges")
    patch_rows: list[dict[str, object]] = []
    while True:
        components = _factor_components(current, n)
        if len(components) == 1:
            return canonical_tour(components[0]), tuple(patch_rows)
        proposals: list[
            tuple[int, Edge, Edge, Edge, Edge]
        ] = []
        for left_index, left in enumerate(components):
            left_edges = tuple(
                edge(left[index], left[(index + 1) % len(left)])
                for index in range(len(left))
            )
            for right in components[left_index + 1 :]:
                right_edges = tuple(
                    edge(right[index], right[(index + 1) % len(right)])
                    for index in range(len(right))
                )
                for old_left in left_edges:
                    a, b = old_left
                    for old_right in right_edges:
                        c, d = old_right
                        for new_left, new_right in (
                            (edge(a, c), edge(b, d)),
                            (edge(a, d), edge(b, c)),
                        ):
                            if new_left not in costs or new_right not in costs:
                                continue
                            delta = (
                                costs[new_left]
                                + costs[new_right]
                                - costs[old_left]
                                - costs[old_right]
                            )
                            proposals.append(
                                (
                                    int(delta),
                                    new_left,
                                    new_right,
                                    old_left,
                                    old_right,
                                )
                            )
        if not proposals:
            raise ValueError("candidate graph cannot patch the selected 2-factor")
        chosen = min(proposals)
        delta, new_left, new_right, old_left, old_right = chosen
        before_components = len(components)
        current.remove(old_left)
        current.remove(old_right)
        current.add(new_left)
        current.add(new_right)
        after_components = len(_factor_components(current, n))
        if after_components != before_components - 1:
            raise RuntimeError("factor patch did not merge exactly two components")
        patch_rows.append(
            {
                "delta": delta,
                "removed_edges": tuple(sorted((old_left, old_right))),
                "added_edges": tuple(sorted((new_left, new_right))),
                "components_before": before_components,
                "components_after": after_components,
            }
        )


def _read_candidate_costs(payload: bytes) -> dict[Edge, int]:
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    costs: dict[Edge, int] = {}
    for row in rows:
        candidate = edge(int(row["u"]), int(row["v"]))
        value = int(row["objective_cost"])
        if candidate in costs:
            raise ValueError(f"duplicate candidate edge: {candidate}")
        costs[candidate] = value
    return _normalize_costs(costs)


def reconstruct_lin318_t0(
    archive_path: str | Path,
    *,
    expected_candidate_count: int = 1500,
    expected_cost: int = 42210,
    expected_candidate_sha256: str | None = TARGET_CANDIDATE_SHA256,
    candidate_member: str = TARGET_CANDIDATE_MEMBER,
    factor_member: str = SOURCE_FACTOR_MEMBER,
) -> dict[str, object]:
    """Reconstruct the strict-1,500 local optimum from the Colab archive."""
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as handle:
        candidate_payload = handle.read(candidate_member)
        factor_payload = json.loads(handle.read(factor_member).decode("utf-8"))
    candidate_sha256 = hashlib.sha256(candidate_payload).hexdigest()
    if (
        expected_candidate_sha256 is not None
        and candidate_sha256 != expected_candidate_sha256
    ):
        raise ValueError(
            "candidate snapshot SHA-256 mismatch: "
            f"{candidate_sha256} != {expected_candidate_sha256}"
        )
    costs = _read_candidate_costs(candidate_payload)
    if len(costs) != int(expected_candidate_count):
        raise ValueError(
            f"candidate count mismatch: {len(costs)} != {expected_candidate_count}"
        )
    selected = factor_payload.get("result", {}).get("selected_edges")
    if not isinstance(selected, list) or not selected:
        raise ValueError("source audit has no selected 2-factor")
    factor = frozenset(edge(int(u), int(v)) for u, v in selected)
    n = _vertex_count(costs)
    if len(factor) != n or not factor.issubset(costs):
        raise ValueError("source 2-factor is not a spanning subset of the target graph")
    source_components = _factor_components(factor, n)
    patched_tour, patches = _patch_factor(factor, costs)
    patched_cost = tour_cost(patched_tour, costs)
    descent = strict_two_three_opt(patched_tour, costs)
    final_tour = tuple(int(vertex) for vertex in descent["tour"])
    final_cost = int(descent["final_cost"])
    if final_cost != int(expected_cost):
        raise RuntimeError(
            f"reconstructed incumbent cost {final_cost}, expected {expected_cost}"
        )
    final_validation = validate_tour(final_tour, costs)
    return {
        "schema": "lin318-strict1500-t0-reconstruction-v1",
        "archive_path": archive.name,
        "candidate_member": candidate_member,
        "candidate_sha256": candidate_sha256,
        "candidate_edge_count": len(costs),
        "edge_costs": costs,
        "factor_member": factor_member,
        "source_factor_edge_count": len(factor),
        "source_factor_component_count": len(source_components),
        "patches": patches,
        "patched_tour": patched_tour,
        "patched_cost": patched_cost,
        "descent": descent,
        "tour": final_tour,
        "cost": final_cost,
        "validation": final_validation,
        "invariants": {
            "candidate_hash_verified": (
                expected_candidate_sha256 is None
                or candidate_sha256 == expected_candidate_sha256
            ),
            "candidate_count_verified": len(costs)
            == int(expected_candidate_count),
            "factor_is_spanning_2regular": True,
            "factor_subset_of_target_candidates": factor.issubset(costs),
            "patch_reduced_to_one_cycle": True,
            "strict_descent_nonincreasing": final_cost <= patched_cost,
            "expected_cost_verified": final_cost == int(expected_cost),
            "final_hamiltonian": bool(final_validation["hamiltonian"]),
            "final_candidate_membership": bool(
                final_validation["candidate_membership"]
            ),
        },
    }


__all__ = [
    "AtomicMove",
    "SOURCE_FACTOR_MEMBER",
    "TARGET_CANDIDATE_MEMBER",
    "TARGET_CANDIDATE_SHA256",
    "canonical_tour",
    "edge",
    "forced_edge_bundle_relaxation",
    "forced_edge_relaxation",
    "minimum_barrier_insertion",
    "reconstruct_lin318_t0",
    "strict_two_three_opt",
    "tour_cost",
    "tour_edge_set",
    "validate_tour",
]
