from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_lin318_threshold_basin import run_lin318_threshold_basin


class Lin318ThresholdBasinRunnerTests(unittest.TestCase):
    ARCHIVE = ROOT / "data" / "lin318_reproduction_inputs.zip"
    SCAN = ROOT / "results" / "lin318" / "closure_scan"

    @unittest.skipUnless(
        ARCHIVE.is_file() and (SCAN / "closure_scan.csv").is_file(),
        "committed LIN318 threshold inputs unavailable",
    )
    def test_reproduces_42118_and_42108_without_truth_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result = run_lin318_threshold_basin(
                self.ARCHIVE,
                self.SCAN,
                Path(raw) / "basin",
            )
            self.assertEqual(result["results"]["t0_cost"], 42_210)
            self.assertEqual(
                result["results"]["best_closure_seed_cost"],
                42_118,
            )
            self.assertEqual(
                result["results"]["best_second_stage_cost"],
                42_108,
            )
            self.assertEqual(
                result["results"]["best_nonpositive_top32_cost"],
                42_108,
            )
            self.assertFalse(
                result["truth_isolation"]["reference_or_optimum_tour_read"]
            )
            self.assertTrue(all(result["hard_invariants"].values()))
            for output in result["csv_outputs"].values():
                self.assertTrue((Path(raw) / "basin" / output).is_file())


if __name__ == "__main__":
    unittest.main()
