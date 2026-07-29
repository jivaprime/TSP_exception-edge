from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_public_results import audit_public_results


class PublicArtifactTests(unittest.TestCase):
    def test_compact_release_audit(self) -> None:
        result = audit_public_results()
        self.assertEqual(result["candidate_graph"]["weak_edges"], 988)
        self.assertEqual(result["candidate_graph"]["target_edges"], 1500)
        self.assertEqual(
            result["threshold_scan"]["safe_positive_lower_bounds"],
            13,
        )
        self.assertEqual(
            result["published_tours"]["best_safe_second_stage"],
            42_108,
        )
        self.assertEqual(result["zero_base_pilot"]["no_cycle_found"], 15)
        self.assertEqual(
            result["exact_small"]["stage3_natural_core"]["safe_mandatory"],
            48,
        )
        self.assertTrue(result["restricted_baseline"]["exact"])
        self.assertFalse(result["optimum_tour_read"])


if __name__ == "__main__":
    unittest.main()
