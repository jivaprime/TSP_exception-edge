"""Audit the compact public artifacts without using an optimum tour.

The default audit checks the sealed candidate snapshots, threshold arithmetic,
published tours, and the exact weak-Delaunay restricted bound.  ``--full``
also reruns the deterministic threshold-to-basin experiment from the compact
bundle and the 45 retained closure witnesses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence
import zipfile

from exception_edge.lin318_basin_escape import (
    TARGET_CANDIDATE_MEMBER,
    reconstruct_lin318_t0,
    tour_cost,
    validate_tour,
)
from run_lin318_threshold_basin import run_lin318_threshold_basin


ROOT = Path(__file__).resolve().parent
CHECKSUMS = ROOT / "checksums.sha256"
ARCHIVE = ROOT / "data" / "lin318_reproduction_inputs.zip"
SCAN_DIR = ROOT / "results" / "lin318" / "closure_scan"
TOUR_DIR = ROOT / "results" / "lin318" / "tours"
RESTRICTED_BOUNDS = (
    ROOT
    / "results"
    / "lin318"
    / "restricted_baseline"
    / "restricted_bounds.csv"
)
PILOT_SUMMARY = ROOT / "results" / "lin318" / "zero_base_pilot_summary.csv"
STAGE2_FAMILY = (
    ROOT
    / "results"
    / "exact_small"
    / "tables"
    / "stage2_closure_family_summary.csv"
)
STAGE3_GROUP = (
    ROOT
    / "results"
    / "exact_small"
    / "tables"
    / "stage3_threshold_group_summary.csv"
)

WEAK_MEMBER = (
    "benchmark_pilot/runs/weak_only__r01/solver_output/"
    "candidate_edges/round_00.csv"
)
NEAREST_MEMBER = (
    "benchmark_pilot/runs/static_nearest_b512__r01/solver_output/"
    "candidate_edges/round_00.csv"
)
EXPECTED_TARGET_SHA256 = (
    "aa75561a53100148eb69b45fef04bd690e5b0f7a0db9d8a573b85ecb1d3d4a69"
)
EXPECTED_WEAK_COUNT = 988
EXPECTED_TARGET_COUNT = 1500
EXPECTED_ADDED_COUNT = 512
EXPECTED_SAFE_COUNT = 13
EXPECTED_RETAINED_WITNESSES = 45
EXPECTED_BASELINE_VALUE = 42_231


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_candidate_costs(payload: bytes) -> dict[tuple[int, int], int]:
    text = payload.decode("utf-8-sig")
    rows = csv.DictReader(text.splitlines())
    result: dict[tuple[int, int], int] = {}
    for row in rows:
        u = int(row["u"])
        v = int(row["v"])
        pair = (u, v) if u < v else (v, u)
        result[pair] = int(row["objective_cost"])
    return result


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _audit_release_checksums() -> int:
    expected: dict[str, str] = {}
    for raw in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, relative = raw.split("  ", 1)
        if relative in expected:
            raise RuntimeError(f"duplicate checksum entry: {relative}")
        expected[relative] = digest

    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != CHECKSUMS
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual_files != set(expected):
        missing = sorted(actual_files - set(expected))
        stale = sorted(set(expected) - actual_files)
        raise RuntimeError(
            f"checksum manifest membership mismatch; missing={missing}, stale={stale}"
        )
    for relative, digest in expected.items():
        payload = (ROOT / relative).read_bytes()
        if _sha256(payload) != digest:
            raise RuntimeError(f"checksum mismatch: {relative}")
    return len(expected)


def audit_public_results(*, full: bool = False) -> dict[str, Any]:
    checksum_count = _audit_release_checksums()
    with zipfile.ZipFile(ARCHIVE) as archive:
        weak_payload = archive.read(WEAK_MEMBER)
        target_payload = archive.read(TARGET_CANDIDATE_MEMBER)
        nearest_payload = archive.read(NEAREST_MEMBER)

    weak_costs = _read_candidate_costs(weak_payload)
    target_costs = _read_candidate_costs(target_payload)
    if len(weak_costs) != EXPECTED_WEAK_COUNT:
        raise RuntimeError(f"weak candidate count changed: {len(weak_costs)}")
    if len(target_costs) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(f"target candidate count changed: {len(target_costs)}")
    if len(set(target_costs) - set(weak_costs)) != EXPECTED_ADDED_COUNT:
        raise RuntimeError("added candidate count changed")
    if _sha256(target_payload) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("target candidate seal changed")
    if target_payload != nearest_payload:
        raise RuntimeError(
            "the published local and nearest candidate snapshots no longer match"
        )

    reconstruction = reconstruct_lin318_t0(ARCHIVE)
    if reconstruction["cost"] != 42_210:
        raise RuntimeError("the sealed T0 reconstruction no longer costs 42,210")

    with (SCAN_DIR / "closure_scan.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        scan_rows = list(csv.DictReader(handle))
    if len(scan_rows) != EXPECTED_ADDED_COUNT:
        raise RuntimeError(f"closure scan row count changed: {len(scan_rows)}")

    verified = 0
    positive = 0
    for row in scan_rows:
        if row["status"] != "verified_closure_witness":
            continue
        verified += 1
        edge_cost = int(row["forced_edge_cost"])
        path_upper = int(row["path_upper"])
        closure_upper = int(row["closure_upper"])
        safe_gain = float(row["safe_gain_lower"])
        safe_kappa = float(row["safe_kappa_lower"])
        crossed = _as_bool(row["safe_threshold_crossed"])

        if closure_upper != path_upper + edge_cost:
            raise RuntimeError("closure/path arithmetic mismatch")
        if safe_gain != EXPECTED_BASELINE_VALUE - closure_upper:
            raise RuntimeError("safe gain arithmetic mismatch")
        expected_kappa = (EXPECTED_BASELINE_VALUE - path_upper) / edge_cost
        if abs(safe_kappa - expected_kappa) > 1e-12:
            raise RuntimeError("safe kappa arithmetic mismatch")
        if crossed != (safe_gain > 0.0):
            raise RuntimeError("threshold flag mismatch")
        positive += int(crossed)

    if verified != EXPECTED_ADDED_COUNT:
        raise RuntimeError(f"verified witness count changed: {verified}")
    if positive != EXPECTED_SAFE_COUNT:
        raise RuntimeError(f"safe-positive count changed: {positive}")

    retained = list((SCAN_DIR / "edges").glob("*/*result.json"))
    if len(retained) != EXPECTED_RETAINED_WITNESSES:
        raise RuntimeError(f"retained witness count changed: {len(retained)}")

    published_tours: dict[str, int] = {}
    for path in sorted(TOUR_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tour = tuple(int(vertex) for vertex in payload["tour_zero_based"])
        validation = validate_tour(tour, target_costs)
        cost = tour_cost(tour, target_costs)
        if not validation["hamiltonian"]:
            raise RuntimeError(f"{path.name} is not Hamiltonian")
        if not validation["candidate_membership"]:
            raise RuntimeError(f"{path.name} leaves the 1,500-edge graph")
        if cost != int(payload["cost"]):
            raise RuntimeError(f"{path.name} cost mismatch")
        published_tours[payload["name"]] = cost

    expected_tours = {
        "t0_strict1500": 42_210,
        "best_safe_closure_seed": 42_118,
        "best_safe_second_stage": 42_108,
        "best_nonpositive_top32_probe": 42_108,
    }
    if published_tours != expected_tours:
        raise RuntimeError(f"published tour set changed: {published_tours}")

    with RESTRICTED_BOUNDS.open(newline="", encoding="utf-8-sig") as handle:
        bounds = {
            row["baseline"]: row for row in csv.DictReader(handle)
        }
    weak_bound = bounds["weak_delaunay_union"]
    lower = int(weak_bound["direct_solver_integer_lower_bound"])
    upper = int(weak_bound["verified_feasible_upper_bound"])
    if (lower, upper, weak_bound["restricted_exact"]) != (
        EXPECTED_BASELINE_VALUE,
        EXPECTED_BASELINE_VALUE,
        "True",
    ):
        raise RuntimeError("weak-Delaunay restricted exact certificate changed")

    with PILOT_SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        pilot_rows = list(csv.DictReader(handle))
    if len(pilot_rows) != 15:
        raise RuntimeError(f"zero-base pilot condition count changed: {len(pilot_rows)}")
    if any(row["final_layer_solver_status"] != "no_cycle_found" for row in pilot_rows):
        raise RuntimeError("the published zero-base pilot outcomes changed")

    with STAGE2_FAMILY.open(newline="", encoding="utf-8-sig") as handle:
        stage2_rows = [
            row
            for row in csv.DictReader(handle)
            if row["objective_mode"] == "euclidean_raw"
        ]
    stage2_total_mandatory = sum(
        int(row["mandatory_exception_count"]) for row in stage2_rows
    )
    nontargeted = [
        row
        for row in stage2_rows
        if row["family"] not in {"hoey5", "hoey5_jitter"}
    ]
    stage2_nontargeted_instances = sum(
        int(row["instance_count"]) for row in nontargeted
    )
    stage2_nontargeted_mandatory = sum(
        int(row["mandatory_exception_count"]) for row in nontargeted
    )
    if (
        stage2_total_mandatory,
        stage2_nontargeted_instances,
        stage2_nontargeted_mandatory,
    ) != (178, 360, 23):
        raise RuntimeError("Stage 2 headline table no longer matches the release")

    with STAGE3_GROUP.open(newline="", encoding="utf-8-sig") as handle:
        stage3_row = next(
            row
            for row in csv.DictReader(handle)
            if row["analysis_group"] == "natural_core"
            and row["objective_mode"] == "euclidean_raw"
            and row["split"] == "validation"
        )
    stage3_headline = {
        "instances": int(stage3_row["instance_count"]),
        "mandatory": int(stage3_row["mandatory_exception_count"]),
        "exact_global_q1": int(
            stage3_row["mandatory_q1_global_explained_count"]
        ),
        "exact_exclusive_q1": int(
            stage3_row["mandatory_q1_exclusive_explained_count"]
        ),
        "safe_mandatory": int(stage3_row["mandatory_safe_positive_count"]),
        "exact_beneficial_pairs": int(
            stage3_row["edge_micro_exact_beneficial_count"]
        ),
        "safe_beneficial_pairs": int(
            stage3_row["edge_micro_certified_beneficial_count"]
        ),
        "all_pairs": int(stage3_row["nonbaseline_edge_total"]),
        "safe_support_superset": int(stage3_row["safe_q1_candidate_count"]),
        "exact_support": int(stage3_row["exact_q1_support_edge_count"]),
    }
    if stage3_headline != {
        "instances": 600,
        "mandatory": 51,
        "exact_global_q1": 51,
        "exact_exclusive_q1": 51,
        "safe_mandatory": 48,
        "exact_beneficial_pairs": 98,
        "safe_beneficial_pairs": 88,
        "all_pairs": 15_054,
        "safe_support_superset": 644,
        "exact_support": 600,
    }:
        raise RuntimeError("Stage 3 headline table no longer matches the release")

    full_reproduction: dict[str, int] | None = None
    if full:
        with tempfile.TemporaryDirectory() as raw:
            reproduced = run_lin318_threshold_basin(
                ARCHIVE,
                SCAN_DIR,
                Path(raw) / "basin",
            )
        full_reproduction = {
            "t0": int(reproduced["results"]["t0_cost"]),
            "closure_seed": int(
                reproduced["results"]["best_closure_seed_cost"]
            ),
            "second_stage": int(
                reproduced["results"]["best_second_stage_cost"]
            ),
            "nonpositive_top32": int(
                reproduced["results"]["best_nonpositive_top32_cost"]
            ),
        }

    return {
        "schema": "exception-edge-public-audit-v1",
        "release_checksums_verified": checksum_count,
        "candidate_graph": {
            "weak_edges": len(weak_costs),
            "target_edges": len(target_costs),
            "added_edges": len(set(target_costs) - set(weak_costs)),
            "target_sha256": _sha256(target_payload),
            "local_equals_nearest_snapshot": True,
        },
        "threshold_scan": {
            "rows": len(scan_rows),
            "verified_witnesses": verified,
            "safe_positive_lower_bounds": positive,
            "nonpositive_rows_are_inconclusive": True,
        },
        "restricted_baseline": {
            "lower": lower,
            "upper": upper,
            "exact": lower == upper,
        },
        "zero_base_pilot": {
            "conditions": len(pilot_rows),
            "no_cycle_found": len(pilot_rows),
        },
        "exact_small": {
            "stage2_raw_mandatory": stage2_total_mandatory,
            "stage2_nontargeted_instances": stage2_nontargeted_instances,
            "stage2_nontargeted_mandatory": stage2_nontargeted_mandatory,
            "stage3_natural_core": stage3_headline,
        },
        "published_tours": published_tours,
        "full_reproduction": full_reproduction,
        "optimum_tour_read": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="also rerun the deterministic 42,210 -> 42,118 -> 42,108 pipeline",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            audit_public_results(full=args.full),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
