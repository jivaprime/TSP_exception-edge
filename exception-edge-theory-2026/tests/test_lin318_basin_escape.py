from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exception_edge.lin318_basin_escape import (
    TARGET_CANDIDATE_SHA256,
    edge,
    forced_edge_bundle_relaxation,
    forced_edge_relaxation,
    minimum_barrier_insertion,
    reconstruct_lin318_t0,
    strict_two_three_opt,
    tour_cost,
    tour_edge_set,
    validate_tour,
)


def _complete_costs(
    n: int,
    *,
    default: int = 20,
    overrides: dict[tuple[int, int], int] | None = None,
) -> dict[tuple[int, int], int]:
    values = {
        (u, v): int(default)
        for u in range(n)
        for v in range(u + 1, n)
    }
    for pair, value in (overrides or {}).items():
        values[edge(*pair)] = int(value)
    return values


class BasinEscapeUnitTests(unittest.TestCase):
    def test_strict_two_opt_and_added_edge_lock(self) -> None:
        costs = _complete_costs(
            4,
            overrides={
                (0, 1): 10,
                (1, 2): 1,
                (2, 3): 10,
                (0, 3): 1,
                (0, 2): 1,
                (1, 3): 1,
            },
        )
        start = (0, 1, 2, 3)
        free = strict_two_three_opt(start, costs)
        self.assertEqual(free["initial_cost"], 22)
        self.assertEqual(free["final_cost"], 4)
        self.assertEqual(free["two_opt_moves"], 1)
        self.assertIn(edge(0, 2), tour_edge_set(free["tour"]))

        locked = strict_two_three_opt(
            start,
            costs,
            locked_edges=(edge(0, 1),),
        )
        self.assertIn(edge(0, 1), tour_edge_set(locked["tour"]))
        self.assertGreaterEqual(locked["final_cost"], free["final_cost"])
        self.assertTrue(locked["validation"]["locked_edges_preserved"])

    def test_minimum_barrier_two_opt_inserts_target(self) -> None:
        costs = _complete_costs(
            4,
            overrides={
                (0, 1): 10,
                (1, 2): 1,
                (2, 3): 10,
                (0, 3): 1,
                (0, 2): 1,
                (1, 3): 1,
            },
        )
        start = (0, 1, 2, 3)
        move = minimum_barrier_insertion(
            start,
            (0, 2),
            costs,
            move_kinds=("2opt",),
        )
        self.assertEqual(move.kind, "2opt")
        self.assertEqual(move.delta, -18)
        self.assertIn(edge(0, 2), move.added_edges)
        self.assertEqual(
            tour_cost(move.resulting_tour, costs),
            tour_cost(start, costs) + move.delta,
        )

    def test_genuine_three_opt_insertion_is_hamiltonian(self) -> None:
        costs = _complete_costs(6, default=1)
        start = tuple(range(6))
        move = minimum_barrier_insertion(
            start,
            (0, 2),
            costs,
            move_kinds=("3opt",),
        )
        self.assertEqual(move.kind, "3opt")
        self.assertEqual(move.delta, 0)
        self.assertIn(edge(0, 2), move.added_edges)
        validation = validate_tour(move.resulting_tour, costs)
        self.assertTrue(validation["hamiltonian"])
        self.assertTrue(validation["candidate_membership"])

    def test_forced_relaxation_preserves_unit_until_release(self) -> None:
        costs = _complete_costs(
            4,
            overrides={
                (0, 1): 10,
                (1, 2): 1,
                (2, 3): 10,
                (0, 3): 1,
                (0, 2): 1,
                (1, 3): 1,
            },
        )
        result = forced_edge_relaxation(
            (0, 1, 2, 3),
            (0, 2),
            costs,
            move_kinds=("2opt",),
        )
        self.assertEqual(result["initial_cost"], 22)
        self.assertEqual(result["final_cost"], 4)
        self.assertTrue(result["escaped_to_better_basin"])
        self.assertTrue(result["target_survives_locked_relaxation"])
        self.assertTrue(result["invariants"]["added_set_preserved_while_locked"])
        self.assertTrue(result["invariants"]["kick_delta_exact"])

    def test_bundle_relaxation_preserves_all_targets_until_release(self) -> None:
        costs = _complete_costs(6, default=1)
        result = forced_edge_bundle_relaxation(
            tuple(range(6)),
            ((0, 2), (3, 5)),
            costs,
            move_kinds=("2opt", "3opt"),
        )
        self.assertTrue(
            result["invariants"]["all_targets_present_before_locked_descent"]
        )
        self.assertTrue(
            result["invariants"]["all_locks_preserved_during_locked_descent"]
        )
        self.assertTrue(result["invariants"]["final_hamiltonian"])
        self.assertTrue(result["invariants"]["final_candidate_membership"])


class Lin318ArchiveIntegrationTests(unittest.TestCase):
    ARCHIVE = ROOT / "data" / "lin318_reproduction_inputs.zip"

    @unittest.skipUnless(ARCHIVE.is_file(), "local Colab result ZIP unavailable")
    def test_reconstructs_strict_1500_incumbent(self) -> None:
        result = reconstruct_lin318_t0(self.ARCHIVE)
        self.assertEqual(result["candidate_edge_count"], 1500)
        self.assertEqual(result["candidate_sha256"], TARGET_CANDIDATE_SHA256)
        self.assertEqual(result["patched_cost"], 42234)
        self.assertEqual(result["cost"], 42210)
        self.assertEqual(result["descent"]["two_opt_moves"], 2)
        self.assertEqual(result["descent"]["three_opt_moves"], 0)
        self.assertTrue(all(result["invariants"].values()))


if __name__ == "__main__":
    unittest.main()
