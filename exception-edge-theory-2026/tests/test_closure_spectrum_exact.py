from __future__ import annotations

from itertools import permutations
from math import sqrt
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exception_edge.exact import (
    held_karp_cycle,
    held_karp_cycle_by_exception_count,
)
from exception_edge.geometry import (
    delaunay_edges,
    distance_matrix,
    tour_edges,
    tour_length,
)


def _brute_force_spectrum(
    dist: np.ndarray,
    baseline_edges: set[tuple[int, int]],
) -> dict[int, tuple[float, list[int]]]:
    """Independent canonical-cycle enumeration for small test instances."""
    n = len(dist)
    result: dict[int, tuple[float, list[int]]] = {}
    for tail in permutations(range(1, n)):
        if tail[0] > tail[-1]:
            continue
        tour = [0, *tail]
        count = len(tour_edges(tour) - baseline_edges)
        value = tour_length(tour, dist)
        previous = result.get(count)
        if (
            previous is None
            or value < previous[0]
            or (value == previous[0] and tour < previous[1])
        ):
            result[count] = (value, tour)
    return dict(sorted(result.items()))


class ClosureSpectrumExactTests(unittest.TestCase):
    def test_spectrum_matches_brute_force_enumeration(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [1.2, 0.1],
                [1.8, 0.9],
                [1.0, 1.7],
                [-0.2, 1.2],
                [0.45, 0.65],
            ]
        )
        dist = distance_matrix(points)
        baseline = {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 4),
            (0, 5),
            (2, 5),
            (4, 5),
        }

        expected = _brute_force_spectrum(dist, baseline)
        actual = held_karp_cycle_by_exception_count(
            dist,
            [
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
                (4, 0),
                (5, 0),
                (5, 2),
                (5, 4),
                (0, 1),
            ],
        )

        self.assertEqual(set(actual), set(expected))
        for count in expected:
            self.assertAlmostEqual(
                actual[count][0], expected[count][0], places=12
            )
            self.assertEqual(actual[count][1], expected[count][1])
            self.assertEqual(
                len(tour_edges(actual[count][1]) - baseline),
                count,
            )
            self.assertAlmostEqual(
                tour_length(actual[count][1], dist),
                actual[count][0],
                places=12,
            )

        reordered = held_karp_cycle_by_exception_count(
            dist, list(reversed(sorted(baseline)))
        )
        self.assertEqual(actual, reordered)

    def test_path_graph_has_topological_exception_number_one(self) -> None:
        n = 7
        points = np.column_stack(
            (np.arange(n, dtype=float), np.zeros(n, dtype=float))
        )
        dist = distance_matrix(points)
        path = {(vertex, vertex + 1) for vertex in range(n - 1)}

        spectrum = held_karp_cycle_by_exception_count(dist, path)

        self.assertNotIn(0, spectrum)
        self.assertEqual(min(spectrum), 1)
        value, witness = spectrum[1]
        self.assertEqual(tour_edges(witness) - path, {(0, n - 1)})
        self.assertAlmostEqual(value, 2.0 * (n - 1), places=12)

    def test_hoey_five_point_delaunay_witness_has_z1_below_z0(
        self,
    ) -> None:
        # Hoey's A=(0,0), B=(4,1), C=(5,2), D=(4,0), E=(1,0)
        # example.  Its Euclidean optimum uses the non-Delaunay edge AB.
        points = np.array(
            [
                [0.0, 0.0],
                [4.0, 1.0],
                [5.0, 2.0],
                [4.0, 0.0],
                [1.0, 0.0],
            ]
        )
        dist = distance_matrix(points)
        baseline = delaunay_edges(points)

        self.assertNotIn((0, 1), baseline)
        spectrum = held_karp_cycle_by_exception_count(dist, baseline)

        self.assertEqual(set(spectrum), {0, 1, 2})
        self.assertAlmostEqual(
            spectrum[0][0],
            5.0 + sqrt(2.0) + sqrt(29.0),
            places=12,
        )
        self.assertAlmostEqual(
            spectrum[1][0],
            4.0 + sqrt(17.0) + sqrt(2.0) + sqrt(5.0),
            places=12,
        )
        self.assertAlmostEqual(
            spectrum[2][0],
            7.0 + sqrt(17.0) + sqrt(2.0) + 2.0 * sqrt(5.0),
            places=12,
        )
        self.assertLess(spectrum[1][0], spectrum[0][0])
        self.assertLess(spectrum[1][0], spectrum[2][0])
        self.assertEqual(
            tour_edges(spectrum[1][1]) - baseline,
            {(0, 1)},
        )

    def test_regular_convex_control_has_z0_as_global_optimum(self) -> None:
        n = 7
        angles = 2.0 * np.pi * np.arange(n) / n
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        dist = distance_matrix(points)
        baseline = delaunay_edges(points)

        spectrum = held_karp_cycle_by_exception_count(dist, baseline)
        optimum, _ = held_karp_cycle(dist)

        self.assertIn(0, spectrum)
        self.assertAlmostEqual(spectrum[0][0], optimum, places=12)
        self.assertAlmostEqual(
            min(value for value, _ in spectrum.values()),
            spectrum[0][0],
            places=12,
        )
        self.assertEqual(
            tour_edges(spectrum[0][1]),
            {
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (0, 6),
            },
        )

    def test_invalid_baseline_edges_are_rejected(self) -> None:
        dist = distance_matrix(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        invalid = (
            [(0, 0)],
            [(0, 3)],
            [(0, 1.5)],
            [(0, 1, 2)],
        )
        for baseline in invalid:
            with self.subTest(baseline=baseline):
                with self.assertRaises(ValueError):
                    held_karp_cycle_by_exception_count(dist, baseline)

    def test_nearly_symmetric_matrix_is_rejected(self) -> None:
        dist = np.ones((4, 4), dtype=float)
        np.fill_diagonal(dist, 0.0)
        dist[0, 1] += 5e-13

        with self.assertRaisesRegex(ValueError, "exactly symmetric"):
            held_karp_cycle_by_exception_count(dist, set())


if __name__ == "__main__":
    unittest.main()
