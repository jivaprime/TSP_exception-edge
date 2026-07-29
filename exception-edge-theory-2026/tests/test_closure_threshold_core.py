from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exception_edge.closure_spectrum import analyze_closure_spectrum
from exception_edge.closure_threshold import (
    analyze_pair_closure_thresholds,
    held_karp_all_endpoint_paths,
)
from exception_edge.geometry import delaunay_edges, distance_matrix


class ClosureThresholdCoreTests(unittest.TestCase):
    def test_pair_closure_identity_on_hoey_fixture(self) -> None:
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
        paths = held_karp_all_endpoint_paths(dist, baseline)
        spectrum = analyze_closure_spectrum(dist, baseline)

        self.assertNotIn((0, 1), baseline)
        self.assertTrue(paths[(0, 1)].feasible)
        self.assertAlmostEqual(
            paths[(0, 1)].value + dist[0, 1],
            spectrum.level(1).value,
            places=12,
        )

    def test_safe_gain_interval_contains_exact_gain(self) -> None:
        rng = np.random.default_rng(20260729)
        for _ in range(8):
            points = rng.uniform(size=(8, 2))
            analysis = analyze_pair_closure_thresholds(
                distance_matrix(points),
                delaunay_edges(points),
                held_karp_bound_iterations=20,
                held_karp_path_bound_iterations=20,
            )
            self.assertLessEqual(
                analysis.cycle_held_karp_lower,
                analysis.cycle_exact + 1e-12,
            )
            for pair in analysis.pairs:
                self.assertLessEqual(
                    pair.gain_lower,
                    pair.gain_exact + 1e-12,
                )
                self.assertLessEqual(
                    pair.gain_exact,
                    pair.gain_upper + 1e-12,
                )

    def test_safe_candidate_set_preserves_exact_q1_support(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [4.0, 1.0],
                [5.0, 2.0],
                [4.0, 0.0],
                [1.0, 0.0],
            ]
        )
        analysis = analyze_pair_closure_thresholds(
            distance_matrix(points),
            delaunay_edges(points),
        )
        z_one = min(pair.closure_exact for pair in analysis.pairs)
        z_one_upper = min(
            pair.path_upper + pair.edge_cost for pair in analysis.pairs
        )
        exact_support = {
            pair.edge
            for pair in analysis.pairs
            if abs(pair.closure_exact - z_one) <= 1e-12
        }
        safe_superset = {
            pair.edge
            for pair in analysis.pairs
            if pair.path_lower + pair.edge_cost <= z_one_upper + 1e-12
        }

        self.assertEqual(exact_support, {(0, 1)})
        self.assertLessEqual(exact_support, safe_superset)


if __name__ == "__main__":
    unittest.main()
