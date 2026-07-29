from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exception_edge.lin318_threshold_escape import (
    SEMANTICS,
    load_candidate_archive,
    normalize_edge,
    prepare_lkh_closure_task,
    run_lkh_forced_closure,
    scan_lkh_forced_closures,
    verify_forced_closure_tour,
)
from exception_edge.tsplib_io import load_euc_2d_instance


def _write_instance(path: Path, n: int = 6):
    points = [
        (1, 0, 0),
        (2, 10, 0),
        (3, 20, 0),
        (4, 20, 10),
        (5, 10, 10),
        (6, 0, 10),
    ][:n]
    path.write_text(
        "\n".join(
            [
                "NAME : tiny",
                "TYPE : TSP",
                f"DIMENSION : {n}",
                "EDGE_WEIGHT_TYPE : EUC_2D",
                "NODE_COORD_SECTION",
                *[f"{node} {x} {y}" for node, x, y in points],
                "EOF",
                "",
            ]
        ),
        encoding="ascii",
    )
    return load_euc_2d_instance(path)


def _all_edges(n: int) -> set[tuple[int, int]]:
    return {(u, v) for u in range(n) for v in range(u + 1, n)}


def _tour_text(tour: list[int]) -> str:
    return "\n".join(
        [
            "NAME : fake",
            "TYPE : TOUR",
            f"DIMENSION : {len(tour)}",
            "TOUR_SECTION",
            *[str(vertex + 1) for vertex in tour],
            "-1",
            "EOF",
            "",
        ]
    )


def _snapshot_payload(edges: set[tuple[int, int]]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=["u", "v", "objective_cost"],
        lineterminator="\n",
    )
    writer.writeheader()
    for u, v in sorted(edges):
        writer.writerow({"u": u, "v": v, "objective_cost": 1})
    return stream.getvalue().encode("utf-8")


def _write_archive(
    path: Path,
    baseline: set[tuple[int, int]],
    target: set[tuple[int, int]],
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for run_id, edges in (
            ("weak_only__r01", baseline),
            ("static_local_b512__r01", target),
        ):
            payload = _snapshot_payload(edges)
            prefix = (
                f"benchmark_pilot/runs/{run_id}/solver_output"
            )
            archive.writestr(
                f"{prefix}/candidate_edges/round_00.csv",
                payload,
            )
            archive.writestr(
                f"{prefix}/blind_solver_seal.json",
                json.dumps(
                    {
                        "initial_candidate_edges_sha256": (
                            hashlib.sha256(payload).hexdigest()
                        )
                    }
                ),
            )


class EdgeAndArchiveTests(unittest.TestCase):
    def test_normalize_edge_guards_endpoints(self) -> None:
        self.assertEqual(normalize_edge(4, 1, n_vertices=5), (1, 4))
        with self.assertRaises(ValueError):
            normalize_edge(2, 2)
        with self.assertRaises(ValueError):
            normalize_edge(-1, 2)
        with self.assertRaises(ValueError):
            normalize_edge(0, 5, n_vertices=5)
        with self.assertRaises(ValueError):
            normalize_edge(True, 2)

    def test_archive_loading_checks_seals_and_difference(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = {(0, 1), (1, 2), (2, 3)}
            target = baseline | {(0, 2), (1, 3)}
            archive_path = root / "results.zip"
            _write_archive(archive_path, baseline, target)

            selection = load_candidate_archive(
                archive_path,
                n_vertices=4,
                expected_target_edge_count=5,
                expected_added_edge_count=2,
            )
            self.assertEqual(selection.baseline_edges, frozenset(baseline))
            self.assertEqual(
                selection.added_edges,
                frozenset({(0, 2), (1, 3)}),
            )
            self.assertEqual(len(selection.archive_sha256), 64)


class ForcedClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.instance = _write_instance(self.root / "tiny.tsp")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_problem_contains_penalties_and_fixed_edge(self) -> None:
        forced = (0, 2)
        baseline = _all_edges(6) - {forced}
        task = prepare_lkh_closure_task(
            self.instance,
            baseline,
            forced,
            self.root / "task",
            penalty=1000,
            max_trials=12,
        )
        problem = task.problem_path.read_text(encoding="ascii")
        parameters = task.parameter_path.read_text(encoding="ascii")
        self.assertIn("FIXED_EDGES_SECTION\n1 3\n-1", problem)
        self.assertIn("EDGE_WEIGHT_FORMAT : FULL_MATRIX", problem)
        self.assertIn("MAX_TRIALS = 12", parameters)
        self.assertIn("TOUR_FILE = forced_closure.tour", parameters)

    def test_valid_witness_yields_only_a_safe_lower(self) -> None:
        forced = (0, 2)
        baseline = _all_edges(6) - {forced}
        tour = (0, 2, 1, 4, 3, 5)
        result = verify_forced_closure_tour(
            self.instance,
            baseline,
            forced,
            tour,
            baseline_cycle_lower_bound=100.0,
            baseline_lower_bound_provenance="unit-test bound",
        )
        self.assertEqual(result["semantics"], SEMANTICS)
        self.assertFalse(result["exact_kappa_computed"])
        self.assertFalse(result["exact_hamiltonian_path_solved"])
        self.assertEqual(
            result["closure_upper"],
            result["path_upper"] + result["forced_edge_cost"],
        )
        self.assertEqual(
            result["safe_gain_lower"],
            100.0 - result["closure_upper"],
        )
        self.assertEqual(
            (result["path_witness"][0], result["path_witness"][-1]),
            forced,
        )

    def test_independent_verifier_rejects_an_outside_edge(self) -> None:
        forced = (0, 2)
        other_missing = (1, 3)
        baseline = _all_edges(6) - {forced, other_missing}
        tour_using_both = (0, 2, 4, 5, 1, 3)
        with self.assertRaisesRegex(ValueError, "outside baseline"):
            verify_forced_closure_tour(
                self.instance,
                baseline,
                forced,
                tour_using_both,
                baseline_cycle_lower_bound=0.0,
                baseline_lower_bound_provenance="unit-test bound",
            )

    def test_run_parses_and_verifies_lkh_output(self) -> None:
        forced = (0, 2)
        baseline = _all_edges(6) - {forced}
        fake_tour = [0, 2, 1, 4, 3, 5]

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            (cwd / "forced_closure.tour").write_text(
                _tour_text(fake_tour),
                encoding="ascii",
            )
            return subprocess.CompletedProcess(command, 0, "ok", "")

        with patch(
            "exception_edge.lin318_threshold_escape.subprocess.run",
            side_effect=fake_run,
        ):
            result = run_lkh_forced_closure(
                self.instance,
                baseline,
                forced,
                lkh_executable=sys.executable,
                task_dir=self.root / "run",
                baseline_cycle_lower_bound=100.0,
                baseline_lower_bound_provenance="unit-test bound",
            )
        self.assertEqual(result["status"], "verified_closure_witness")
        self.assertTrue(result["hamiltonicity_verified"])
        self.assertEqual(result["lkh_returncode"], 0)
        self.assertFalse((self.root / "run/forced_closure.tsp").exists())
        self.assertFalse((self.root / "run/forced_closure.tour").exists())


class ResumeParallelScanTests(unittest.TestCase):
    def test_scan_runs_in_parallel_and_resumes_without_lkh(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instance = _write_instance(root / "tiny.tsp")
            candidates = {(0, 2), (1, 3)}
            baseline = _all_edges(6) - candidates
            tours = {
                (0, 2): [0, 2, 1, 4, 3, 5],
                (1, 3): [1, 3, 0, 4, 2, 5],
            }

            def fake_run(command, **kwargs):
                cwd = Path(kwargs["cwd"])
                name = cwd.name
                forced = tuple(int(value) for value in name.split("_"))
                (cwd / "forced_closure.tour").write_text(
                    _tour_text(tours[forced]),
                    encoding="ascii",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            output = root / "scan"
            with patch(
                "exception_edge.lin318_threshold_escape.subprocess.run",
                side_effect=fake_run,
            ) as mocked:
                first = scan_lkh_forced_closures(
                    instance,
                    baseline,
                    candidates,
                    lkh_executable=sys.executable,
                    output_dir=output,
                    baseline_cycle_lower_bound=100.0,
                    baseline_lower_bound_provenance="unit-test bound",
                    workers=2,
                )
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(first["executed_this_call_count"], 2)
            self.assertEqual(first["verified_closure_witness_count"], 2)
            self.assertTrue((output / "closure_scan.csv").is_file())
            self.assertTrue((output / "scan_summary.json").is_file())
            self.assertFalse(
                (
                    output
                    / "edges/000000_000002/forced_closure.tsp"
                ).exists()
            )
            saved_result = json.loads(
                (
                    output / "edges/000000_000002/result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_result["tour"], tours[(0, 2)])
            self.assertEqual(len(saved_result["path_witness"]), 6)

            with patch(
                "exception_edge.lin318_threshold_escape.subprocess.run",
                side_effect=AssertionError("resume must not invoke LKH"),
            ) as resumed_mock:
                second = scan_lkh_forced_closures(
                    instance,
                    baseline,
                    candidates,
                    lkh_executable=sys.executable,
                    output_dir=output,
                    baseline_cycle_lower_bound=100.0,
                    baseline_lower_bound_provenance="unit-test bound",
                    workers=2,
                )
            self.assertEqual(resumed_mock.call_count, 0)
            self.assertEqual(second["resumed_count"], 2)
            self.assertEqual(second["executed_this_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
