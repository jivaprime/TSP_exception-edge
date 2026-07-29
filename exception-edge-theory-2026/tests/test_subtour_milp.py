from __future__ import annotations

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exception_edge.subtour_milp import solve_hamiltonian_cycle


class SubtourMILPTests(unittest.TestCase):
    def test_iterative_secs_recover_exact_optimal_tour(self) -> None:
        # The minimum degree-two solution is two cost-3 triangles.  SECs force
        # a Hamiltonian cycle containing two cross edges, with value 24.
        costs = {
            (u, v): (
                1.0
                if (u < 3 and v < 3) or (u >= 3 and v >= 3)
                else 10.0
            )
            for u in range(6)
            for v in range(u + 1, 6)
        }
        result = solve_hamiltonian_cycle(
            6,
            costs,
            objective_upper_bound=24.0,
            max_iterations=10,
        )

        self.assertEqual(result.status, "optimal")
        self.assertTrue(result.exact)
        self.assertAlmostEqual(result.objective, 24.0)
        self.assertIsNotNone(result.lower_bound)
        self.assertAlmostEqual(result.lower_bound, 24.0)
        self.assertIsNotNone(result.cycle)
        self.assertEqual(set(result.cycle), set(range(6)))
        self.assertEqual(len(result.selected_edges), 6)
        self.assertGreaterEqual(result.iterations, 2)
        self.assertEqual(result.cuts, len(result.sec_subsets))
        self.assertIn(result.sec_subsets[0], ((0, 1, 2), (3, 4, 5)))
        self.assertEqual(len(result.rounds), result.iterations)
        self.assertEqual(result.rounds[0].component_sizes, (3, 3))
        self.assertGreaterEqual(result.wall_seconds, 0.0)
        self.assertTrue(
            all(round_result.wall_seconds >= 0.0 for round_result in result.rounds)
        )
        self.assertTrue(
            all(
                round_result.mip_node_count is None
                or round_result.mip_node_count >= 0
                for round_result in result.rounds
            )
        )

    def test_disconnected_graph_has_no_hamiltonian_cycle(self) -> None:
        costs = {
            (0, 1): 1.0,
            (0, 2): 1.0,
            (1, 2): 1.0,
            (3, 4): 1.0,
            (3, 5): 1.0,
            (4, 5): 1.0,
        }
        result = solve_hamiltonian_cycle(6, costs, max_iterations=10)

        self.assertEqual(result.status, "infeasible")
        self.assertFalse(result.exact)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.cycle)
        self.assertEqual(result.selected_edges, ())
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.cuts, 1)
        self.assertIn(result.sec_subsets[0], ((0, 1, 2), (3, 4, 5)))
        self.assertEqual(len(result.rounds), result.iterations)

    def test_valid_secs_can_be_reused_after_candidate_expansion(self) -> None:
        costs = {
            (u, v): (
                1.0
                if (u < 3 and v < 3) or (u >= 3 and v >= 3)
                else 10.0
            )
            for u in range(6)
            for v in range(u + 1, 6)
        }
        first = solve_hamiltonian_cycle(6, costs, max_iterations=10)
        reused = solve_hamiltonian_cycle(
            6,
            costs,
            max_iterations=10,
            initial_sec_subsets=first.sec_subsets,
        )

        self.assertTrue(first.exact)
        self.assertTrue(reused.exact)
        self.assertEqual(reused.objective, first.objective)
        self.assertGreaterEqual(reused.rounds[0].cuts_before, 1)
        self.assertLessEqual(reused.iterations, first.iterations)

    def test_invalid_initial_sec_is_rejected(self) -> None:
        costs = {
            (u, v): 1.0
            for u in range(5)
            for v in range(u + 1, 5)
        }
        with self.assertRaisesRegex(ValueError, "between 2 and n-2"):
            solve_hamiltonian_cycle(
                5,
                costs,
                initial_sec_subsets=[(0,)],
            )

    def test_feasible_limit_is_retained_until_cutoff_proves_optimality(self) -> None:
        costs = {
            (u, v): 1.0
            for u in range(4)
            for v in range(u + 1, 4)
        }
        edges = tuple(sorted(costs))
        cycle_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
        first_x = np.asarray(
            [1.0 if candidate in cycle_edges else 0.0 for candidate in edges]
        )
        responses = [
            SimpleNamespace(
                status=1,
                message="time limit",
                fun=4.0,
                mip_dual_bound=3.0,
                mip_node_count=1,
                mip_gap=0.25,
                x=first_x,
            ),
            SimpleNamespace(
                status=2,
                message="infeasible below incumbent",
                fun=None,
                mip_dual_bound=None,
                mip_node_count=0,
                mip_gap=None,
                x=None,
            ),
        ]
        with patch(
            "exception_edge.subtour_milp.milp",
            side_effect=responses,
        ) as mocked:
            result = solve_hamiltonian_cycle(
                4,
                costs,
                max_iterations=3,
                continue_after_feasible_limit=True,
                objective_granularity=1.0,
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result.status, "optimal_cutoff")
        self.assertTrue(result.exact)
        self.assertEqual(result.objective, 4.0)
        self.assertEqual(result.lower_bound, 4.0)
        self.assertIsNotNone(result.cycle)
        self.assertIsNotNone(result.first_feasible_wall_seconds_upper_bound)
        self.assertIsNotNone(result.best_feasible_wall_seconds_upper_bound)
        self.assertLessEqual(
            result.best_feasible_wall_seconds_upper_bound,
            result.wall_seconds,
        )


if __name__ == "__main__":
    unittest.main()
