"""Reproduce the sealed LIN318 threshold-to-basin experiment.

The runner has three deliberately separate roles:

1. reconstruct the strict 1,500-edge incumbent ``T0`` from the Colab archive;
2. use only the safe-positive forced-closure witnesses as local-search seeds
   and interventions;
3. probe the 32 best *nonpositive/inconclusive* rows from the initial closure
   scan while preserving every safe-positive edge that survives in the best
   stage-two tour during the locked part of the intervention.

No reference or optimum tour is opened.  The expected costs below are
reproduction invariants for already observed, selection-blind experiment
states; they are not used to select edges or to guide local search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from exception_edge.lin318_basin_escape import (
    Edge,
    TARGET_CANDIDATE_MEMBER,
    TARGET_CANDIDATE_SHA256,
    edge,
    forced_edge_relaxation,
    minimum_barrier_insertion,
    reconstruct_lin318_t0,
    strict_two_three_opt,
    tour_cost,
    tour_edge_set,
    validate_tour,
)


EXPECTED_SAFE_POSITIVE_COUNT = 13
EXPECTED_T0_COST = 42_210
EXPECTED_BEST_CLOSURE_SEED_COST = 42_118
EXPECTED_BEST_CLOSURE_SEED_EDGE = (88, 98)
EXPECTED_BEST_SECOND_STAGE_COST = 42_108
EXPECTED_BEST_SECOND_STAGE_EDGE = (27, 102)
NONPOSITIVE_PROBE_COUNT = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tour_sha256(tour: Sequence[int]) -> str:
    payload = ",".join(str(int(vertex)) for vertex in tour).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (tuple, list, dict, set, frozenset)):
                    value = json.dumps(
                        _json_ready(value),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                serializable[key] = value
            writer.writerow(serializable)


def _relative_artifact_path(path: Path, root: Path) -> str:
    """Return a stable POSIX path relative to a declared artifact root."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_tour_bundle(
    output: Path,
    name: str,
    tour: Sequence[int],
    costs: Mapping[Edge, int],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = tuple(int(vertex) for vertex in tour)
    validation = validate_tour(normalized, costs)
    cost = int(validation["cost"])
    digest = _tour_sha256(normalized)
    json_path = output / "tours" / f"{name}.json"
    tour_path = output / "tours" / f"{name}.tour"
    _write_json(
        json_path,
        {
            "schema": "lin318-threshold-basin-tour-v1",
            "name": name,
            "cost": cost,
            "tour_sha256": digest,
            "tour_zero_based": normalized,
            "tour_one_based": tuple(vertex + 1 for vertex in normalized),
            "validation": validation,
            "provenance": provenance,
            "reference_or_optimum_tour_read": False,
        },
    )
    tsp_lines = [
        f"NAME : {name}",
        "TYPE : TOUR",
        f"DIMENSION : {len(normalized)}",
        f"COMMENT : candidate-restricted cost {cost}; zero-based SHA256 {digest}",
        "TOUR_SECTION",
        *(str(vertex + 1) for vertex in normalized),
        "-1",
        "EOF",
    ]
    tour_path.parent.mkdir(parents=True, exist_ok=True)
    tour_path.write_text("\n".join(tsp_lines) + "\n", encoding="ascii")
    return {
        "name": name,
        "cost": cost,
        "tour_sha256": digest,
        "json": _relative_artifact_path(json_path, output),
        "tsplib_tour": _relative_artifact_path(tour_path, output),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean field: {value!r}")


def _as_int(value: Any, *, field: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {field}: {value!r}") from exc


def _as_float(value: Any, *, field: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number in {field}: {value!r}") from exc


def _scan_edge(row: Mapping[str, Any]) -> Edge:
    return edge(
        _as_int(row.get("forced_edge_u"), field="forced_edge_u"),
        _as_int(row.get("forced_edge_v"), field="forced_edge_v"),
    )


def _read_scan_rows(scan_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    csv_path = scan_dir / "closure_scan.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("closure scan CSV is empty")
    required = {
        "forced_edge_u",
        "forced_edge_v",
        "status",
        "closure_upper",
        "safe_gain_lower",
        "safe_threshold_crossed",
        "hamiltonicity_verified",
        "forced_edge_verified",
        "baseline_path_membership_verified",
        "objective_recomputed_independently",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"closure scan lacks required columns: {sorted(missing)}")
    seen: set[Edge] = set()
    for row in rows:
        candidate = _scan_edge(row)
        if candidate in seen:
            raise ValueError(f"duplicate scan edge: {candidate}")
        seen.add(candidate)
    return csv_path, rows


def _verified_witness_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "verified_closure_witness"
        and _as_bool(row.get("hamiltonicity_verified"))
        and _as_bool(row.get("forced_edge_verified"))
        and _as_bool(row.get("baseline_path_membership_verified"))
        and _as_bool(row.get("objective_recomputed_independently"))
    )


def _safe_positive(row: Mapping[str, Any]) -> bool:
    crossed = _as_bool(row.get("safe_threshold_crossed"))
    gain = _as_float(row.get("safe_gain_lower"), field="safe_gain_lower")
    if crossed != (gain > 0.0):
        raise ValueError(
            f"threshold flag/gain disagreement for edge {_scan_edge(row)}"
        )
    return _verified_witness_row(row) and crossed


def _index_result_files(scan_dir: Path) -> dict[Edge, tuple[Path, dict[str, Any]]]:
    indexed: dict[Edge, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((scan_dir / "edges").glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = edge(
            _as_int(payload.get("forced_edge_u"), field="forced_edge_u"),
            _as_int(payload.get("forced_edge_v"), field="forced_edge_v"),
        )
        if candidate in indexed:
            raise ValueError(f"duplicate result artifact for edge {candidate}")
        indexed[candidate] = (path, payload)
    if not indexed:
        raise ValueError(f"no per-edge result artifacts below {scan_dir / 'edges'}")
    return indexed


def _verified_closure_tour(
    candidate: Edge,
    row: Mapping[str, Any],
    result_index: Mapping[Edge, tuple[Path, dict[str, Any]]],
    costs: Mapping[Edge, int],
) -> tuple[tuple[int, ...], Path, dict[str, Any]]:
    if not _verified_witness_row(row):
        raise ValueError(f"edge {candidate} does not have a verified closure witness")
    try:
        result_path, payload = result_index[candidate]
    except KeyError as exc:
        raise FileNotFoundError(
            f"missing result.json for closure edge {candidate}"
        ) from exc
    if payload.get("status") != "verified_closure_witness":
        raise ValueError(f"result artifact for {candidate} is not verified")
    tour_payload = payload.get("tour")
    if not isinstance(tour_payload, list) or not tour_payload:
        raise ValueError(f"result artifact for {candidate} has no tour")
    tour = tuple(int(vertex) for vertex in tour_payload)
    validation = validate_tour(tour, costs)
    recomputed = int(validation["cost"])
    csv_cost = _as_int(row.get("closure_upper"), field="closure_upper")
    json_cost = _as_int(payload.get("closure_upper"), field="closure_upper")
    if recomputed != csv_cost or recomputed != json_cost:
        raise RuntimeError(
            f"closure cost disagreement for {candidate}: "
            f"recomputed={recomputed}, csv={csv_cost}, json={json_cost}"
        )
    if candidate not in tour_edge_set(tour):
        raise RuntimeError(f"closure tour for {candidate} lost its forced edge")
    return tour, result_path, payload


def _edge_tag(candidate: Edge) -> str:
    return f"{candidate[0]:06d}_{candidate[1]:06d}"


def _basic_scan_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "closure_upper": _as_int(row.get("closure_upper"), field="closure_upper"),
        "safe_gain_lower": _as_float(
            row.get("safe_gain_lower"), field="safe_gain_lower"
        ),
        "safe_kappa_lower": _as_float(
            row.get("safe_kappa_lower"), field="safe_kappa_lower"
        ),
        "safe_threshold_crossed": _as_bool(row.get("safe_threshold_crossed")),
    }


def _move_columns(result: Mapping[str, Any]) -> dict[str, Any]:
    move = result.get("atomic_move")
    if not isinstance(move, Mapping):
        return {
            "move_kind": "",
            "move_pattern": "",
            "immediate_barrier": "",
            "added_edges": (),
            "removed_edges": (),
        }
    return {
        "move_kind": move.get("kind", ""),
        "move_pattern": move.get("pattern", ""),
        "immediate_barrier": result.get("immediate_barrier", move.get("delta", "")),
        "added_edges": move.get("added_edges", ()),
        "removed_edges": move.get("removed_edges", ()),
    }


def _anchored_forced_relaxation(
    tour: Sequence[int],
    target: Edge,
    costs: Mapping[Edge, int],
    anchors: Iterable[Edge],
) -> dict[str, Any]:
    """Insert ``target`` while preserving anchors and the atomic added set."""
    initial = tuple(int(vertex) for vertex in tour)
    initial_cost = tour_cost(initial, costs)
    normalized_anchors = frozenset(edge(*pair) for pair in anchors)
    initial_edges = tour_edge_set(initial)
    if not normalized_anchors:
        raise ValueError("the anchored probe requires at least one surviving anchor")
    if not normalized_anchors.issubset(initial_edges):
        raise ValueError("one or more declared anchors are absent from the initial tour")
    if target in initial_edges:
        raise ValueError("target is already present; no atomic insertion is defined")
    move = minimum_barrier_insertion(
        initial,
        target,
        costs,
        move_kinds=("2opt", "3opt"),
        locked_edges=normalized_anchors,
    )
    kicked = tuple(int(vertex) for vertex in move.resulting_tour)
    locks = frozenset(set(normalized_anchors).union(move.added_edges))
    kicked_validation = validate_tour(kicked, costs, locked_edges=locks)
    kicked_cost = int(kicked_validation["cost"])
    if kicked_cost != initial_cost + int(move.delta):
        raise RuntimeError("anchored intervention kick delta is not exact")
    locked = strict_two_three_opt(kicked, costs, locked_edges=locks)
    locked_tour = tuple(int(vertex) for vertex in locked["tour"])
    locked_cost = int(locked["final_cost"])
    if not locks.issubset(tour_edge_set(locked_tour)):
        raise RuntimeError("anchored locked relaxation lost a locked edge")
    released = strict_two_three_opt(locked_tour, costs)
    final_tour = tuple(int(vertex) for vertex in released["tour"])
    final_cost = int(released["final_cost"])
    if locked_cost > kicked_cost or final_cost > locked_cost:
        raise RuntimeError("anchored relaxation violated monotone descent")
    final_validation = validate_tour(final_tour, costs)
    final_edges = tour_edge_set(final_tour)
    return {
        "schema": "lin318-anchored-forced-edge-relaxation-v1",
        "target_edge": target,
        "anchor_edges": tuple(sorted(normalized_anchors)),
        "initial_tour": initial,
        "initial_cost": initial_cost,
        "atomic_move": move.result_dict(),
        "immediate_barrier": int(move.delta),
        "kicked_tour": kicked,
        "kicked_cost": kicked_cost,
        "locked_edges": tuple(sorted(locks)),
        "locked_relaxation": locked,
        "locked_cost": locked_cost,
        "unlocked_relaxation": released,
        "final_tour": final_tour,
        "final_cost": final_cost,
        "escape_gain": initial_cost - final_cost,
        "escaped_to_better_basin": final_cost < initial_cost,
        "target_survives_final_relaxation": target in final_edges,
        "surviving_anchors": tuple(
            pair for pair in sorted(normalized_anchors) if pair in final_edges
        ),
        "validation": final_validation,
        "invariants": {
            "anchors_present_initially": normalized_anchors.issubset(initial_edges),
            "target_inserted": target in tour_edge_set(kicked),
            "anchor_and_added_set_locked": locks
            == frozenset(set(normalized_anchors).union(move.added_edges)),
            "all_locks_preserved_during_locked_descent": locks.issubset(
                tour_edge_set(locked_tour)
            ),
            "kick_delta_exact": kicked_cost == initial_cost + int(move.delta),
            "locked_descent_nonincreasing": locked_cost <= kicked_cost,
            "unlocked_descent_nonincreasing": final_cost <= locked_cost,
            "final_hamiltonian": bool(final_validation["hamiltonian"]),
            "final_candidate_membership": bool(
                final_validation["candidate_membership"]
            ),
        },
    }


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty; choose a new directory: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)


def run_lin318_threshold_basin(
    colab_zip: str | Path,
    closure_scan_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_safe_positive_count: int = EXPECTED_SAFE_POSITIVE_COUNT,
    expected_t0_cost: int = EXPECTED_T0_COST,
    expected_best_closure_seed_cost: int = EXPECTED_BEST_CLOSURE_SEED_COST,
    expected_best_closure_seed_edge: Edge = EXPECTED_BEST_CLOSURE_SEED_EDGE,
    expected_best_second_stage_cost: int = EXPECTED_BEST_SECOND_STAGE_COST,
    expected_best_second_stage_edge: Edge = EXPECTED_BEST_SECOND_STAGE_EDGE,
    nonpositive_probe_count: int = NONPOSITIVE_PROBE_COUNT,
) -> dict[str, Any]:
    """Run the sealed LIN318 basin experiment without reading truth artifacts."""
    archive = Path(colab_zip).resolve()
    scan_dir = Path(closure_scan_dir).resolve()
    output = Path(output_dir).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not scan_dir.is_dir():
        raise FileNotFoundError(scan_dir)
    if int(nonpositive_probe_count) <= 0:
        raise ValueError("nonpositive_probe_count must be positive")
    _prepare_output(output)

    scan_csv, scan_rows = _read_scan_rows(scan_dir)
    row_by_edge = {_scan_edge(row): row for row in scan_rows}
    result_index = _index_result_files(scan_dir)
    positives = sorted(
        (candidate, row)
        for candidate, row in row_by_edge.items()
        if _safe_positive(row)
    )
    if len(positives) != int(expected_safe_positive_count):
        raise RuntimeError(
            f"safe-positive count {len(positives)}, "
            f"expected {expected_safe_positive_count}"
        )

    reconstruction = reconstruct_lin318_t0(
        archive,
        expected_cost=int(expected_t0_cost),
    )
    costs = reconstruction["edge_costs"]
    if not isinstance(costs, Mapping):
        raise RuntimeError("reconstruction did not return candidate edge costs")
    t0 = tuple(int(vertex) for vertex in reconstruction["tour"])
    if tour_cost(t0, costs) != int(expected_t0_cost):
        raise RuntimeError("T0 objective invariant failed")
    if len(costs) != 1_500:
        raise RuntimeError("the target candidate graph does not contain 1,500 edges")
    if reconstruction["candidate_sha256"] != TARGET_CANDIDATE_SHA256:
        raise RuntimeError("the target candidate snapshot hash changed")

    tour_artifacts: list[dict[str, Any]] = []
    tour_artifacts.append(
        _write_tour_bundle(
            output,
            "t0_strict1500",
            t0,
            costs,
            provenance={
                "source": "reconstruct_lin318_t0",
                "candidate_member": TARGET_CANDIDATE_MEMBER,
            },
        )
    )
    details = output / "details"

    # Stage A: every safe-positive closure witness becomes an independent
    # coordinate-only, candidate-restricted strict 2/3-opt seed.
    closure_rows: list[dict[str, Any]] = []
    closure_states: dict[Edge, tuple[int, tuple[int, ...]]] = {}
    all_closure_verified = True
    for candidate, scan_row in positives:
        raw_tour, result_path, _ = _verified_closure_tour(
            candidate,
            scan_row,
            result_index,
            costs,
        )
        raw_cost = tour_cost(raw_tour, costs)
        descent = strict_two_three_opt(raw_tour, costs)
        final_tour = tuple(int(vertex) for vertex in descent["tour"])
        final_cost = int(descent["final_cost"])
        final_validation = validate_tour(final_tour, costs)
        if final_cost > raw_cost:
            raise RuntimeError(f"closure-seed descent increased cost for {candidate}")
        closure_states[candidate] = (final_cost, final_tour)
        target_survives = candidate in tour_edge_set(final_tour)
        detail_path = details / "safe_closure" / f"{_edge_tag(candidate)}.json"
        _write_json(
            detail_path,
            {
                "schema": "lin318-safe-closure-seed-descent-v1",
                "target_edge": candidate,
                "scan_fields": _basic_scan_columns(scan_row),
                "closure_result_path": str(result_path),
                "raw_closure_tour": raw_tour,
                "raw_closure_cost": raw_cost,
                "descent": descent,
                "final_validation": final_validation,
                "target_survives_descent": target_survives,
            },
        )
        closure_rows.append(
            {
                "edge_u": candidate[0],
                "edge_v": candidate[1],
                **_basic_scan_columns(scan_row),
                "raw_closure_cost": raw_cost,
                "strict_final_cost": final_cost,
                "strict_gain": raw_cost - final_cost,
                "two_opt_moves": descent["two_opt_moves"],
                "three_opt_moves": descent["three_opt_moves"],
                "target_survives": target_survives,
                "tour_sha256": _tour_sha256(final_tour),
                "detail_json": _relative_artifact_path(detail_path, output),
            }
        )
        all_closure_verified = all_closure_verified and bool(
            final_validation["hamiltonian"]
            and final_validation["candidate_membership"]
        )

    best_seed_edge, (best_seed_cost, best_seed_tour) = min(
        closure_states.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    expected_seed_edge = edge(*expected_best_closure_seed_edge)
    if best_seed_cost != int(expected_best_closure_seed_cost):
        raise RuntimeError(
            f"best closure seed cost {best_seed_cost}, "
            f"expected {expected_best_closure_seed_cost}"
        )
    if best_seed_edge != expected_seed_edge:
        raise RuntimeError(
            f"best closure seed edge {best_seed_edge}, expected {expected_seed_edge}"
        )
    tour_artifacts.append(
        _write_tour_bundle(
            output,
            "best_safe_closure_seed",
            best_seed_tour,
            costs,
            provenance={
                "source": "safe_positive_closure_then_strict_2_3opt",
                "source_edge": best_seed_edge,
            },
        )
    )

    # Stage B0: direct, independent interventions from T0.  An edge already
    # present in T0 is recorded, not spuriously removed and reinserted.
    t0_edges = tour_edge_set(t0)
    direct_rows: list[dict[str, Any]] = []
    for candidate, scan_row in positives:
        detail_path = details / "safe_direct_t0" / f"{_edge_tag(candidate)}.json"
        if candidate in t0_edges:
            payload: dict[str, Any] = {
                "schema": "lin318-safe-direct-t0-v1",
                "status": "already_present",
                "target_edge": candidate,
                "initial_cost": int(expected_t0_cost),
                "final_cost": int(expected_t0_cost),
                "final_tour": t0,
                "target_survives_final_relaxation": True,
            }
        else:
            try:
                relaxation = forced_edge_relaxation(t0, candidate, costs)
                payload = {
                    "schema": "lin318-safe-direct-t0-v1",
                    "status": "forced_relaxation",
                    **relaxation,
                }
            except ValueError as exc:
                payload = {
                    "schema": "lin318-safe-direct-t0-v1",
                    "status": "operator_unreachable",
                    "target_edge": candidate,
                    "initial_cost": int(expected_t0_cost),
                    "error": str(exc),
                }
        _write_json(detail_path, payload)
        direct_rows.append(
            {
                "edge_u": candidate[0],
                "edge_v": candidate[1],
                **_basic_scan_columns(scan_row),
                "status": payload["status"],
                "initial_cost": payload.get("initial_cost", ""),
                "kicked_cost": payload.get("kicked_cost", ""),
                "locked_cost": payload.get("locked_cost", ""),
                "final_cost": payload.get("final_cost", ""),
                "escape_gain": payload.get("escape_gain", 0),
                "target_survives_final": payload.get(
                    "target_survives_final_relaxation", ""
                ),
                **_move_columns(payload),
                "detail_json": _relative_artifact_path(detail_path, output),
            }
        )

    # Stage B1: starting from the best closure seed, test every other
    # safe-positive edge as one independent second intervention.
    seed_edges = tour_edge_set(best_seed_tour)
    second_rows: list[dict[str, Any]] = []
    second_states: dict[Edge, tuple[int, tuple[int, ...]]] = {}
    for candidate, scan_row in positives:
        if candidate == best_seed_edge:
            continue
        detail_path = details / "safe_second_stage" / f"{_edge_tag(candidate)}.json"
        if candidate in seed_edges:
            payload = {
                "schema": "lin318-safe-second-stage-v1",
                "status": "already_present",
                "target_edge": candidate,
                "initial_cost": best_seed_cost,
                "final_cost": best_seed_cost,
                "final_tour": best_seed_tour,
                "escape_gain": 0,
                "target_survives_final_relaxation": True,
            }
        else:
            try:
                relaxation = forced_edge_relaxation(
                    best_seed_tour,
                    candidate,
                    costs,
                )
                payload = {
                    "schema": "lin318-safe-second-stage-v1",
                    "status": "forced_relaxation",
                    **relaxation,
                }
            except ValueError as exc:
                payload = {
                    "schema": "lin318-safe-second-stage-v1",
                    "status": "operator_unreachable",
                    "target_edge": candidate,
                    "initial_cost": best_seed_cost,
                    "error": str(exc),
                }
        _write_json(detail_path, payload)
        if "final_cost" in payload and "final_tour" in payload:
            final_tour = tuple(int(vertex) for vertex in payload["final_tour"])
            final_cost = int(payload["final_cost"])
            validate_tour(final_tour, costs)
            second_states[candidate] = (final_cost, final_tour)
        second_rows.append(
            {
                "edge_u": candidate[0],
                "edge_v": candidate[1],
                **_basic_scan_columns(scan_row),
                "status": payload["status"],
                "initial_cost": payload.get("initial_cost", ""),
                "kicked_cost": payload.get("kicked_cost", ""),
                "locked_cost": payload.get("locked_cost", ""),
                "final_cost": payload.get("final_cost", ""),
                "escape_gain": payload.get("escape_gain", ""),
                "target_survives_final": payload.get(
                    "target_survives_final_relaxation", ""
                ),
                **_move_columns(payload),
                "detail_json": _relative_artifact_path(detail_path, output),
            }
        )
    if not second_states:
        raise RuntimeError("no executable safe-positive second-stage intervention")
    best_second_edge, (best_second_cost, best_second_tour) = min(
        second_states.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    expected_second_edge = edge(*expected_best_second_stage_edge)
    if best_second_cost != int(expected_best_second_stage_cost):
        raise RuntimeError(
            f"best second-stage cost {best_second_cost}, "
            f"expected {expected_best_second_stage_cost}"
        )
    if best_second_edge != expected_second_edge:
        raise RuntimeError(
            f"best second-stage edge {best_second_edge}, "
            f"expected {expected_second_edge}"
        )
    tour_artifacts.append(
        _write_tour_bundle(
            output,
            "best_safe_second_stage",
            best_second_tour,
            costs,
            provenance={
                "source": "best_safe_closure_seed_plus_one_safe_intervention",
                "seed_edge": best_seed_edge,
                "intervention_edge": best_second_edge,
            },
        )
    )

    # Stage C: initial-scan nonpositive rows are not certified negatives.
    # Rank them solely by their pre-existing closure upper bound, then preserve
    # the surviving safe-positive anchors plus every edge added by the atomic
    # insertion during locked descent.
    positive_edge_set = frozenset(candidate for candidate, _ in positives)
    best_second_edges = tour_edge_set(best_second_tour)
    anchors = tuple(sorted(positive_edge_set.intersection(best_second_edges)))
    if not anchors:
        raise RuntimeError("the 42,108 tour has no surviving safe-positive anchor")
    nonpositive = [
        (candidate, row)
        for candidate, row in row_by_edge.items()
        if _verified_witness_row(row) and not _safe_positive(row)
    ]
    nonpositive.sort(
        key=lambda item: (
            _as_int(item[1].get("closure_upper"), field="closure_upper"),
            item[0],
        )
    )
    selected_nonpositive = nonpositive[: int(nonpositive_probe_count)]
    if len(selected_nonpositive) != int(nonpositive_probe_count):
        raise RuntimeError(
            f"only {len(selected_nonpositive)} verified nonpositive rows; "
            f"need {nonpositive_probe_count}"
        )
    probe_rows: list[dict[str, Any]] = []
    probe_states: dict[Edge, tuple[int, tuple[int, ...]]] = {}
    for rank, (candidate, scan_row) in enumerate(selected_nonpositive, start=1):
        # Revalidate the initial scan artifact even though its tour is not used
        # as the intervention seed.
        _, result_path, _ = _verified_closure_tour(
            candidate,
            scan_row,
            result_index,
            costs,
        )
        detail_path = details / "nonpositive_top32" / f"{rank:02d}_{_edge_tag(candidate)}.json"
        if candidate in best_second_edges:
            payload = {
                "schema": "lin318-nonpositive-anchored-probe-v1",
                "status": "already_present",
                "target_edge": candidate,
                "anchor_edges": anchors,
                "initial_cost": best_second_cost,
                "final_cost": best_second_cost,
                "final_tour": best_second_tour,
                "escape_gain": 0,
                "target_survives_final_relaxation": True,
                "surviving_anchors": anchors,
            }
        else:
            try:
                relaxation = _anchored_forced_relaxation(
                    best_second_tour,
                    candidate,
                    costs,
                    anchors,
                )
                payload = {
                    "schema": "lin318-nonpositive-anchored-probe-v1",
                    "status": "anchored_forced_relaxation",
                    **relaxation,
                }
            except ValueError as exc:
                payload = {
                    "schema": "lin318-nonpositive-anchored-probe-v1",
                    "status": "operator_unreachable",
                    "target_edge": candidate,
                    "anchor_edges": anchors,
                    "initial_cost": best_second_cost,
                    "error": str(exc),
                }
        payload["initial_scan_result_path"] = _relative_artifact_path(
            result_path,
            scan_dir,
        )
        _write_json(detail_path, payload)
        if "final_cost" in payload and "final_tour" in payload:
            final_tour = tuple(int(vertex) for vertex in payload["final_tour"])
            final_cost = int(payload["final_cost"])
            validate_tour(final_tour, costs)
            probe_states[candidate] = (final_cost, final_tour)
        probe_rows.append(
            {
                "initial_scan_rank": rank,
                "edge_u": candidate[0],
                "edge_v": candidate[1],
                **_basic_scan_columns(scan_row),
                "status": payload["status"],
                "anchors": anchors,
                "initial_cost": payload.get("initial_cost", ""),
                "kicked_cost": payload.get("kicked_cost", ""),
                "locked_cost": payload.get("locked_cost", ""),
                "final_cost": payload.get("final_cost", ""),
                "escape_gain": payload.get("escape_gain", ""),
                "target_survives_final": payload.get(
                    "target_survives_final_relaxation", ""
                ),
                "surviving_anchors": payload.get("surviving_anchors", ()),
                **_move_columns(payload),
                "detail_json": _relative_artifact_path(detail_path, output),
            }
        )
    if not probe_states:
        raise RuntimeError("none of the nonpositive top-32 probes was executable")
    best_probe_edge, (best_probe_cost, best_probe_tour) = min(
        probe_states.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    tour_artifacts.append(
        _write_tour_bundle(
            output,
            "best_nonpositive_top32_probe",
            best_probe_tour,
            costs,
            provenance={
                "source": "initial_scan_nonpositive_top32_anchored_probe",
                "source_edge": best_probe_edge,
                "safe_positive_anchors": anchors,
            },
        )
    )

    csv_outputs = {
        "safe_positive_closure_seeds": output
        / "safe_positive_closure_seeds.csv",
        "safe_positive_direct_t0": output / "safe_positive_direct_t0.csv",
        "safe_positive_second_stage": output
        / "safe_positive_second_stage.csv",
        "nonpositive_top32_anchored": output
        / "nonpositive_top32_anchored.csv",
    }
    _write_csv(csv_outputs["safe_positive_closure_seeds"], closure_rows)
    _write_csv(csv_outputs["safe_positive_direct_t0"], direct_rows)
    _write_csv(csv_outputs["safe_positive_second_stage"], second_rows)
    _write_csv(csv_outputs["nonpositive_top32_anchored"], probe_rows)

    hard_invariants = {
        "reference_or_optimum_tour_not_read": True,
        "candidate_edge_count_is_1500": len(costs) == 1_500,
        "candidate_sha256_verified": reconstruction["candidate_sha256"]
        == TARGET_CANDIDATE_SHA256,
        "t0_cost_verified": tour_cost(t0, costs) == int(expected_t0_cost),
        "safe_positive_count_verified": len(positives)
        == int(expected_safe_positive_count),
        "all_safe_closure_tours_verified": all_closure_verified,
        "best_closure_seed_cost_verified": best_seed_cost
        == int(expected_best_closure_seed_cost),
        "best_closure_seed_edge_verified": best_seed_edge == expected_seed_edge,
        "best_second_stage_cost_verified": best_second_cost
        == int(expected_best_second_stage_cost),
        "best_second_stage_edge_verified": best_second_edge
        == expected_second_edge,
        "best_second_stage_hamiltonian": bool(
            validate_tour(best_second_tour, costs)["hamiltonian"]
        ),
        "best_second_stage_candidate_membership": bool(
            validate_tour(best_second_tour, costs)["candidate_membership"]
        ),
        "safe_positive_anchor_nonempty": bool(anchors),
        "nonpositive_probe_count_verified": len(selected_nonpositive)
        == int(nonpositive_probe_count),
        "nonpositive_rows_are_not_claimed_certified_negative": True,
    }
    if not all(hard_invariants.values()):
        failed = [key for key, value in hard_invariants.items() if not value]
        raise RuntimeError(f"hard invariants failed: {failed}")

    summary: dict[str, Any] = {
        "schema": "lin318-threshold-basin-reproduction-v1",
        "semantics": {
            "safe_positive": (
                "feasible closure below the declared baseline-cycle lower bound; "
                "a sufficient lower certificate, not exact kappa"
            ),
            "nonpositive": (
                "nonpositive feasible-witness lower bound; inconclusive, not a "
                "certificate that the edge is nonbeneficial"
            ),
            "local_search": (
                "strict candidate-restricted 2-opt plus four genuine 3-opt "
                "reconnections"
            ),
            "intervention": (
                "minimum-barrier atomic insertion; added-set lock; strict locked "
                "descent; full release; strict descent"
            ),
        },
        "inputs": {
            "colab_zip": archive.name,
            "colab_zip_sha256": _sha256_file(archive),
            "candidate_member": TARGET_CANDIDATE_MEMBER,
            "candidate_sha256": reconstruction["candidate_sha256"],
            "closure_scan_dir": scan_dir.name,
            "closure_scan_csv": _relative_artifact_path(scan_csv, scan_dir),
            "closure_scan_csv_sha256": _sha256_file(scan_csv),
        },
        "counts": {
            "candidate_edges": len(costs),
            "scan_rows": len(scan_rows),
            "safe_positive_edges": len(positives),
            "safe_direct_t0_already_present": sum(
                row["status"] == "already_present" for row in direct_rows
            ),
            "safe_direct_t0_forced": sum(
                row["status"] == "forced_relaxation" for row in direct_rows
            ),
            "safe_direct_t0_operator_unreachable": sum(
                row["status"] == "operator_unreachable" for row in direct_rows
            ),
            "safe_second_stage_tests": len(second_rows),
            "nonpositive_top32_tests": len(probe_rows),
            "safe_positive_anchor_count": len(anchors),
        },
        "results": {
            "t0_cost": int(expected_t0_cost),
            "best_closure_seed_edge": best_seed_edge,
            "best_closure_seed_cost": best_seed_cost,
            "best_second_stage_edge": best_second_edge,
            "best_second_stage_cost": best_second_cost,
            "improvement_from_t0_to_best_second_stage": int(expected_t0_cost)
            - best_second_cost,
            "surviving_safe_positive_anchors": anchors,
            "best_nonpositive_top32_edge": best_probe_edge,
            "best_nonpositive_top32_cost": best_probe_cost,
            "best_nonpositive_top32_gain_from_42108": best_second_cost
            - best_probe_cost,
        },
        "csv_outputs": {
            key: _relative_artifact_path(path, output)
            for key, path in csv_outputs.items()
        },
        "tour_outputs": tour_artifacts,
        "hard_invariants": hard_invariants,
        "truth_isolation": {
            "reference_or_optimum_tour_read": False,
            "selection_inputs": [
                "Colab 1,500-edge candidate snapshot and saved 2-factor",
                "initial forced-closure scan rows and their verified tours",
            ],
            "forbidden_selection_inputs_used": [],
        },
    }
    summary_path = output / "run_summary.json"
    _write_json(summary_path, summary)
    return {
        **summary,
        "run_summary_json": _relative_artifact_path(summary_path, output),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce LIN318 safe-threshold closure seeds, forced basin "
            "interventions, and anchored nonpositive top-32 probes."
        )
    )
    parser.add_argument(
        "--colab-zip",
        required=True,
        type=Path,
        help="compact archive, e.g. data/lin318_reproduction_inputs.zip",
    )
    parser.add_argument(
        "--closure-scan",
        required=True,
        type=Path,
        help="initial closure scan directory containing closure_scan.csv and edges/",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new or empty output directory",
    )
    parser.add_argument(
        "--nonpositive-probe-count",
        type=int,
        default=NONPOSITIVE_PROBE_COUNT,
        help=f"number of initial-scan nonpositive rows to probe (default: {NONPOSITIVE_PROBE_COUNT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_lin318_threshold_basin(
        args.colab_zip,
        args.closure_scan,
        args.output,
        nonpositive_probe_count=args.nonpositive_probe_count,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "best_closure_seed_cost": result["results"][
                    "best_closure_seed_cost"
                ],
                "best_second_stage_cost": result["results"][
                    "best_second_stage_cost"
                ],
                "best_nonpositive_top32_cost": result["results"][
                    "best_nonpositive_top32_cost"
                ],
                "run_summary_json": result["run_summary_json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
