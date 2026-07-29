from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from math import inf
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import Delaunay, QhullError

Edge = tuple[int, int]


def edge(u: int, v: int) -> Edge:
    """Return an undirected edge in canonical order."""
    if u == v:
        raise ValueError("self-loops are not valid TSP edges")
    return (u, v) if u < v else (v, u)


def all_edges(n: int) -> list[Edge]:
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


def distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def tour_edges(tour: Sequence[int]) -> set[Edge]:
    if len(tour) < 3:
        raise ValueError("a Hamiltonian cycle needs at least three vertices")
    return {
        edge(int(tour[i]), int(tour[(i + 1) % len(tour)]))
        for i in range(len(tour))
    }


def tour_length(tour: Sequence[int], dist: np.ndarray) -> float:
    return float(
        sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    )


def cross_edges(tour: Sequence[int], side_a: Iterable[int]) -> set[Edge]:
    side_a = set(side_a)
    return {
        e
        for e in tour_edges(tour)
        if ((e[0] in side_a) != (e[1] in side_a))
    }


def delaunay_edges_with_status(points: np.ndarray) -> tuple[set[Edge], bool]:
    """Return planar Delaunay edges and whether a 1-D fallback was used.

    Qhull may reject collinear or nearly duplicated inputs.  In that case the
    function returns consecutive edges of the projection order.  Callers must
    preserve the returned fallback flag because this graph is not comparable
    to an ordinary nondegenerate planar Delaunay triangulation.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return set(), False
    if n == 2:
        return {(0, 1)}, False
    try:
        tri = Delaunay(points, qhull_options="Qbb Qc Qz Q12")
        result: set[Edge] = set()
        for simplex in tri.simplices:
            finite = [int(v) for v in simplex if int(v) < n]
            for u, v in combinations(finite, 2):
                result.add(edge(u, v))
        return result, False
    except QhullError:
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        order = np.argsort(centered @ direction)
        fallback = {
            edge(int(order[i]), int(order[i + 1])) for i in range(n - 1)
        }
        return fallback, True


def delaunay_edges(points: np.ndarray) -> set[Edge]:
    """Return the edge set only; experiments should use the status variant."""
    return delaunay_edges_with_status(points)[0]


class _DisjointSet:
    def __init__(self, vertices: Iterable[int]):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, x: int) -> int:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def mst(
    dist: np.ndarray,
    vertices: Sequence[int] | None = None,
    forced_edge: Edge | None = None,
) -> tuple[float, set[Edge]]:
    """Kruskal MST on an induced complete graph, optionally containing one edge."""
    if vertices is None:
        vertices = list(range(len(dist)))
    vertices = list(map(int, vertices))
    if len(vertices) <= 1:
        if forced_edge is not None:
            raise ValueError("cannot force an edge into a one-vertex MST")
        return 0.0, set()

    vertex_set = set(vertices)
    dsu = _DisjointSet(vertices)
    chosen: set[Edge] = set()
    total = 0.0
    if forced_edge is not None:
        f = edge(*forced_edge)
        if not set(f) <= vertex_set:
            raise ValueError("forced edge is outside the induced vertex set")
        dsu.union(*f)
        chosen.add(f)
        total += float(dist[f])

    candidates = sorted(
        (
            (float(dist[u, v]), edge(u, v))
            for u, v in combinations(vertices, 2)
            if forced_edge is None or edge(u, v) != edge(*forced_edge)
        ),
        key=lambda item: (item[0], item[1]),
    )
    for weight, e in candidates:
        if dsu.union(*e):
            chosen.add(e)
            total += weight
            if len(chosen) == len(vertices) - 1:
                break
    if len(chosen) != len(vertices) - 1:
        raise RuntimeError("induced complete graph unexpectedly disconnected")
    return total, chosen


def diameter(dist: np.ndarray, vertices: Sequence[int]) -> float:
    vertices = list(vertices)
    if len(vertices) <= 1:
        return 0.0
    return max(float(dist[u, v]) for u, v in combinations(vertices, 2))


def separation(
    dist: np.ndarray, side_a: Sequence[int], side_b: Sequence[int]
) -> float:
    return min(float(dist[u, v]) for u in side_a for v in side_b)


def rho_scores(dist: np.ndarray) -> dict[Edge, float]:
    """Raw detour ratio min_w (d(u,w)+d(w,v))/d(u,v)."""
    n = len(dist)
    scores: dict[Edge, float] = {}
    for u, v in all_edges(n):
        direct = float(dist[u, v])
        if direct <= 0:
            scores[(u, v)] = inf
            continue
        detour = min(
            float(dist[u, w] + dist[w, v])
            for w in range(n)
            if w != u and w != v
        )
        scores[(u, v)] = detour / direct
    return scores


def mutual_neighbor_ranks(dist: np.ndarray) -> dict[Edge, int]:
    n = len(dist)
    ranks = np.empty((n, n), dtype=int)
    for u in range(n):
        order = [int(v) for v in np.argsort(dist[u]) if int(v) != u]
        for rank, v in enumerate(order, start=1):
            ranks[u, v] = rank
    return {
        (u, v): int(max(ranks[u, v], ranks[v, u]))
        for u, v in all_edges(n)
    }


def mst_bottleneck_cuts(
    dist: np.ndarray, max_cuts: int = 3, min_side: int = 2
) -> list[tuple[tuple[int, ...], tuple[int, ...], Edge, float]]:
    """Cuts induced by deleting the longest edges of the Euclidean MST."""
    n = len(dist)
    _, tree = mst(dist)
    ranked = sorted(tree, key=lambda e: (-float(dist[e]), e))
    cuts: list[tuple[tuple[int, ...], tuple[int, ...], Edge, float]] = []
    for removed in ranked:
        adjacency: dict[int, list[int]] = defaultdict(list)
        for u, v in tree:
            if (u, v) == removed:
                continue
            adjacency[u].append(v)
            adjacency[v].append(u)
        seen = {removed[0]}
        queue = deque([removed[0]])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        other = set(range(n)) - seen
        if min(len(seen), len(other)) < min_side:
            continue
        a = tuple(sorted(seen))
        b = tuple(sorted(other))
        if a[0] > b[0]:
            a, b = b, a
        cuts.append((a, b, removed, float(dist[removed])))
        if len(cuts) >= max_cuts:
            break
    return cuts
