"""Scalable feasible-path screening for LIN318 closure candidates.

This module deliberately does *not* compute the exact Stage-3 quantity

``kappa_e = (Z_D - H_e) / c(e)``.

For a nonbaseline edge ``e={s,t}``, LKH is asked to construct a Hamiltonian
cycle in ``D + e`` that is forced to contain ``e``.  The returned tour is
independently checked.  Removing ``e`` then gives a feasible baseline-only
Hamiltonian ``s``--``t`` path with cost ``H_upper``.  Given an independently
valid lower bound ``Z_lower <= Z_D``, the module reports only

``safe_gain_lower = Z_lower - H_upper - c(e)``

and

``safe_kappa_lower = (Z_lower - H_upper) / c(e)``.

Consequently, ``safe_gain_lower > 0`` (equivalently
``safe_kappa_lower > 1``) is a proof-safe sufficient condition, conditional
on the supplied cycle lower bound.  A nonpositive value is inconclusive.
Neither a failed LKH run nor failure to find a feasible path is a negative
certificate.

The scan is outcome-isolated: it consumes a TSPLIB problem, sealed candidate
snapshots, an LKH executable, and a declared baseline-cycle lower bound.  It
does not read an optimum value or a reference optimum tour.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from math import isfinite
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from .geometry import Edge, tour_edges
from .tsplib_io import (
    TSPLIBInstance,
    load_euc_2d_instance,
    tour_cost,
    validate_tour,
)


SCHEMA = "lin318-feasible-path-threshold-scan-v1"
RESULT_SCHEMA = "lin318-feasible-path-threshold-result-v1"
SEMANTICS = "feasible_path_safe_lower_not_exact_kappa"


@dataclass(frozen=True)
class CandidateArchiveSelection:
    """Two verified candidate snapshots loaded from a Colab result archive."""

    archive_path: str
    archive_sha256: str
    baseline_run_id: str
    target_run_id: str
    baseline_snapshot_sha256: str
    target_snapshot_sha256: str
    baseline_edges: frozenset[Edge]
    target_edges: frozenset[Edge]
    added_edges: frozenset[Edge]


@dataclass(frozen=True)
class LKHClosureTask:
    """Filesystem artifacts for one forced-edge LKH closure attempt."""

    forced_edge: Edge
    task_dir: Path
    problem_path: Path
    parameter_path: Path
    tour_path: Path
    stdout_path: Path
    stderr_path: Path


def normalize_edge(
    u: int,
    v: int,
    *,
    n_vertices: int | None = None,
) -> Edge:
    """Return a canonical undirected edge and reject malformed endpoints."""

    if isinstance(u, bool) or isinstance(v, bool):
        raise ValueError("edge endpoints must be integers, not booleans")
    try:
        left = int(u)
        right = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError("edge endpoints must be integers") from exc
    if left != u or right != v:
        raise ValueError("edge endpoints must be integers")
    if left == right:
        raise ValueError("self-loops are not valid TSP edges")
    if left < 0 or right < 0:
        raise ValueError("edge endpoints must be nonnegative")
    if n_vertices is not None:
        if n_vertices < 2:
            raise ValueError("n_vertices must be at least two")
        if left >= n_vertices or right >= n_vertices:
            raise ValueError(
                f"edge {(left, right)} is outside range({n_vertices})"
            )
    return (left, right) if left < right else (right, left)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _edge_set_sha256(edges: Iterable[Edge]) -> str:
    payload = "".join(f"{u},{v}\n" for u, v in sorted(set(edges)))
    return _sha256_bytes(payload.encode("ascii"))


def _atomic_json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_member(run_id: str, relative: str) -> str:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError(f"unsafe archive run id: {run_id!r}")
    return f"benchmark_pilot/runs/{run_id}/solver_output/{relative}"


def _read_snapshot_from_archive(
    archive: zipfile.ZipFile,
    run_id: str,
    *,
    n_vertices: int | None,
) -> tuple[frozenset[Edge], str]:
    member = _candidate_member(run_id, "candidate_edges/round_00.csv")
    seal_member = _candidate_member(run_id, "blind_solver_seal.json")
    try:
        payload = archive.read(member)
        seal_payload = archive.read(seal_member)
    except KeyError as exc:
        raise ValueError(
            f"archive is missing the sealed candidate artifact {exc}"
        ) from exc

    snapshot_sha256 = _sha256_bytes(payload)
    try:
        seal = json.loads(seal_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed blind seal for {run_id}") from exc
    declared = str(seal.get("initial_candidate_edges_sha256", ""))
    if declared != snapshot_sha256:
        raise ValueError(
            f"candidate snapshot hash mismatch for {run_id}: "
            f"declared={declared!r}, observed={snapshot_sha256!r}"
        )

    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    except UnicodeError as exc:
        raise ValueError(f"candidate snapshot is not UTF-8: {member}") from exc
    if reader.fieldnames is None or not {"u", "v"} <= set(reader.fieldnames):
        raise ValueError(f"candidate snapshot lacks u/v columns: {member}")

    edges: set[Edge] = set()
    row_count = 0
    for row_count, row in enumerate(reader, start=1):
        try:
            candidate = normalize_edge(
                int(row["u"]),
                int(row["v"]),
                n_vertices=n_vertices,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"malformed candidate edge in {member}, row {row_count + 1}"
            ) from exc
        if candidate in edges:
            raise ValueError(
                f"duplicate candidate edge {candidate} in {member}"
            )
        edges.add(candidate)
    if row_count == 0:
        raise ValueError(f"candidate snapshot is empty: {member}")
    return frozenset(edges), snapshot_sha256


def load_candidate_archive(
    archive_path: str | Path,
    *,
    baseline_run_id: str = "weak_only__r01",
    target_run_id: str = "static_local_b512__r01",
    n_vertices: int | None = None,
    expected_target_edge_count: int | None = 1500,
    expected_added_edge_count: int | None = 512,
) -> CandidateArchiveSelection:
    """Load and hash-check baseline and target candidate snapshots.

    The target snapshot must contain the baseline snapshot.  The returned
    ``added_edges`` are the only edges to which the original q=1 closure
    threshold applies when the baseline is the weak-Delaunay union.
    """

    path = Path(archive_path)
    if not path.is_file():
        raise ValueError(f"candidate archive does not exist: {path}")
    archive_sha256 = _sha256_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            baseline, baseline_sha = _read_snapshot_from_archive(
                archive,
                baseline_run_id,
                n_vertices=n_vertices,
            )
            target, target_sha = _read_snapshot_from_archive(
                archive,
                target_run_id,
                n_vertices=n_vertices,
            )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {path}") from exc

    if not baseline <= target:
        missing = sorted(baseline - target)
        raise ValueError(
            "target candidate graph does not contain its declared baseline; "
            f"first missing edges={missing[:5]}"
        )
    added = frozenset(target - baseline)
    if (
        expected_target_edge_count is not None
        and len(target) != int(expected_target_edge_count)
    ):
        raise ValueError(
            f"target candidate count is {len(target)}, expected "
            f"{expected_target_edge_count}"
        )
    if (
        expected_added_edge_count is not None
        and len(added) != int(expected_added_edge_count)
    ):
        raise ValueError(
            f"nonbaseline candidate count is {len(added)}, expected "
            f"{expected_added_edge_count}"
        )
    return CandidateArchiveSelection(
        archive_path=path.name,
        archive_sha256=archive_sha256,
        baseline_run_id=baseline_run_id,
        target_run_id=target_run_id,
        baseline_snapshot_sha256=baseline_sha,
        target_snapshot_sha256=target_sha,
        baseline_edges=baseline,
        target_edges=target,
        added_edges=added,
    )


def _normalize_edge_set(
    edges: Iterable[Edge],
    *,
    n_vertices: int,
    label: str,
) -> frozenset[Edge]:
    try:
        normalized = frozenset(
            normalize_edge(u, v, n_vertices=n_vertices) for u, v in edges
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a malformed edge") from exc
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _validate_positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    converted = int(value)
    if converted != value or converted <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return converted


def prepare_lkh_closure_task(
    instance: TSPLIBInstance,
    baseline_edges: Iterable[Edge],
    forced_edge: Edge,
    task_dir: str | Path,
    *,
    penalty: int = 2_000_000,
    runs: int = 1,
    max_trials: int | None = None,
    seed: int = 20260729,
) -> LKHClosureTask:
    """Write one explicit-matrix LKH problem with a forced closure edge."""

    n = instance.dimension
    baseline = _normalize_edge_set(
        baseline_edges,
        n_vertices=n,
        label="baseline_edges",
    )
    forced = normalize_edge(*forced_edge, n_vertices=n)
    if forced in baseline:
        raise ValueError("forced closure edge must be outside the baseline")
    penalty_value = _validate_positive_int(penalty, label="penalty")
    run_count = _validate_positive_int(runs, label="runs")
    trial_count = _validate_positive_int(
        n if max_trials is None else max_trials,
        label="max_trials",
    )
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")

    directory = Path(task_dir)
    directory.mkdir(parents=True, exist_ok=True)
    task = LKHClosureTask(
        forced_edge=forced,
        task_dir=directory,
        problem_path=directory / "forced_closure.tsp",
        parameter_path=directory / "forced_closure.par",
        tour_path=directory / "forced_closure.tour",
        stdout_path=directory / "lkh.stdout.txt",
        stderr_path=directory / "lkh.stderr.txt",
    )
    allowed = baseline | {forced}
    matrix_lines: list[str] = []
    for u in range(n):
        row: list[str] = []
        for v in range(n):
            if u == v:
                value = 0
            else:
                candidate = normalize_edge(u, v)
                base_cost = int(instance.distances[u, v])
                value = (
                    base_cost
                    if candidate in allowed
                    else base_cost + penalty_value
                )
            row.append(str(value))
        matrix_lines.append(" ".join(row))

    problem_lines = [
        f"NAME : forced_closure_{forced[0] + 1}_{forced[1] + 1}",
        "TYPE : TSP",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
        *matrix_lines,
        "FIXED_EDGES_SECTION",
        f"{forced[0] + 1} {forced[1] + 1}",
        "-1",
        "EOF",
    ]
    task.problem_path.write_text(
        "\n".join(problem_lines) + "\n",
        encoding="ascii",
    )
    parameter_lines = [
        f"PROBLEM_FILE = {task.problem_path.name}",
        f"TOUR_FILE = {task.tour_path.name}",
        f"RUNS = {run_count}",
        f"MAX_TRIALS = {trial_count}",
        f"SEED = {int(seed)}",
        "TRACE_LEVEL = 0",
    ]
    task.parameter_path.write_text(
        "\n".join(parameter_lines) + "\n",
        encoding="ascii",
    )
    return task


def _parse_lkh_index_tour(path: Path, n_vertices: int) -> tuple[int, ...]:
    if not path.is_file():
        raise ValueError("LKH did not create a tour file")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read LKH tour file: {path}") from exc

    in_section = False
    terminated = False
    identifiers: list[int] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "TOUR_SECTION":
            if in_section or identifiers:
                raise ValueError("duplicate TOUR_SECTION")
            in_section = True
            continue
        if not in_section:
            continue
        for token in line.split():
            try:
                identifier = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"invalid LKH tour token on line {line_number}: {token!r}"
                ) from exc
            if identifier == -1:
                terminated = True
                in_section = False
                break
            identifiers.append(identifier)
        if terminated:
            break
    if not terminated:
        raise ValueError("LKH TOUR_SECTION is absent or lacks -1")
    expected = set(range(1, n_vertices + 1))
    actual = set(identifiers)
    if len(identifiers) != n_vertices or actual != expected:
        raise ValueError(
            "LKH tour is not a permutation of generated node identifiers"
        )
    return tuple(identifier - 1 for identifier in identifiers)


def _path_after_removing_forced_edge(
    tour: Sequence[int],
    forced_edge: Edge,
) -> tuple[int, ...]:
    normalized = tuple(int(vertex) for vertex in tour)
    forced = normalize_edge(*forced_edge, n_vertices=len(normalized))
    location: int | None = None
    for index, current in enumerate(normalized):
        following = normalized[(index + 1) % len(normalized)]
        if normalize_edge(current, following) == forced:
            if location is not None:
                raise ValueError("forced edge occurs more than once in tour")
            location = index
    if location is None:
        raise ValueError("LKH tour does not contain the forced edge")

    # Skip the forced adjacency and traverse every remaining cycle edge.
    path = tuple(
        normalized[(location + 1 + offset) % len(normalized)]
        for offset in range(len(normalized))
    )
    if path[0] != forced[0]:
        path = tuple(reversed(path))
    if path[0] != forced[0] or path[-1] != forced[1]:
        raise AssertionError("forced-edge removal produced wrong endpoints")
    return path


def verify_forced_closure_tour(
    instance: TSPLIBInstance,
    baseline_edges: Iterable[Edge],
    forced_edge: Edge,
    tour: Sequence[int],
    *,
    baseline_cycle_lower_bound: float,
    baseline_lower_bound_provenance: str,
) -> dict[str, Any]:
    """Independently verify a forced closure tour and derive safe lower scores."""

    n = instance.dimension
    baseline = _normalize_edge_set(
        baseline_edges,
        n_vertices=n,
        label="baseline_edges",
    )
    forced = normalize_edge(*forced_edge, n_vertices=n)
    if forced in baseline:
        raise ValueError("forced closure edge must be outside the baseline")
    lower = float(baseline_cycle_lower_bound)
    if not isfinite(lower) or lower < 0.0:
        raise ValueError(
            "baseline_cycle_lower_bound must be finite and nonnegative"
        )
    provenance = str(baseline_lower_bound_provenance).strip()
    if not provenance:
        raise ValueError("baseline_lower_bound_provenance must not be empty")

    normalized_tour = validate_tour(instance, tour)
    selected = tour_edges(normalized_tour)
    if forced not in selected:
        raise ValueError("tour does not contain the forced closure edge")
    outside = selected - (baseline | {forced})
    if outside:
        raise ValueError(
            "tour contains edges outside baseline plus forced edge: "
            f"{sorted(outside)[:5]}"
        )
    path = _path_after_removing_forced_edge(normalized_tour, forced)
    path_selected = {
        normalize_edge(path[index], path[index + 1])
        for index in range(len(path) - 1)
    }
    if not path_selected <= baseline:
        raise AssertionError("forced-edge removal left a nonbaseline path edge")
    if len(path_selected) != n - 1:
        raise AssertionError("closure witness did not produce a simple path")

    closure_upper = int(tour_cost(instance, normalized_tour))
    forced_cost = int(instance.distances[forced])
    path_upper = int(
        sum(
            int(instance.distances[path[index], path[index + 1]])
            for index in range(n - 1)
        )
    )
    if closure_upper != path_upper + forced_cost:
        raise AssertionError("closure and path costs do not add up")
    safe_gain_lower = float(lower - path_upper - forced_cost)
    safe_kappa_lower = (
        float((lower - path_upper) / forced_cost)
        if forced_cost > 0
        else None
    )
    threshold_crossed = bool(safe_gain_lower > 0.0)

    return {
        "schema": RESULT_SCHEMA,
        "semantics": SEMANTICS,
        "status": "verified_closure_witness",
        "forced_edge_u": forced[0],
        "forced_edge_v": forced[1],
        "forced_edge_node_u": instance.node_ids[forced[0]],
        "forced_edge_node_v": instance.node_ids[forced[1]],
        "forced_edge_cost": forced_cost,
        "tour": list(normalized_tour),
        "path_witness": list(path),
        "closure_upper": closure_upper,
        "path_upper": path_upper,
        "declared_baseline_cycle_lower_bound": lower,
        "baseline_lower_bound_provenance": provenance,
        "safe_gain_lower": safe_gain_lower,
        "safe_kappa_lower": safe_kappa_lower,
        "safe_threshold_crossed": threshold_crossed,
        "conditional_certificate": (
            "safe only if the declared baseline-cycle lower bound is valid"
        ),
        "exact_hamiltonian_path_solved": False,
        "exact_kappa_computed": False,
        "hamiltonicity_verified": True,
        "forced_edge_verified": True,
        "baseline_path_membership_verified": True,
        "objective_recomputed_independently": True,
    }


def run_lkh_forced_closure(
    instance: TSPLIBInstance,
    baseline_edges: Iterable[Edge],
    forced_edge: Edge,
    *,
    lkh_executable: str | Path,
    task_dir: str | Path,
    baseline_cycle_lower_bound: float,
    baseline_lower_bound_provenance: str,
    penalty: int = 2_000_000,
    runs: int = 1,
    max_trials: int | None = None,
    seed: int = 20260729,
    timeout_seconds: float = 60.0,
    retain_inputs: bool = False,
) -> dict[str, Any]:
    """Run and independently audit one forced-edge LKH closure attempt."""

    executable = Path(lkh_executable)
    if not executable.is_file():
        raise ValueError(f"LKH executable does not exist: {executable}")
    timeout = float(timeout_seconds)
    if not isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    task = prepare_lkh_closure_task(
        instance,
        baseline_edges,
        forced_edge,
        task_dir,
        penalty=penalty,
        runs=runs,
        max_trials=max_trials,
        seed=seed,
    )

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        if not retain_inputs:
            # Both files are deterministically regenerated from the scan
            # contract.  The verified tour/path order is retained in
            # result.json by the caller.
            task.problem_path.unlink(missing_ok=True)
            task.tour_path.unlink(missing_ok=True)
        return result

    # A partial previous process must never be mistaken for this invocation.
    task.tour_path.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable.resolve()), task.parameter_path.name],
            cwd=task.task_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        task.stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        task.stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout or ""
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr or ""
        )
        task.stdout_path.write_text(stdout, encoding="utf-8")
        task.stderr_path.write_text(stderr, encoding="utf-8")
        return finish({
            "schema": RESULT_SCHEMA,
            "semantics": SEMANTICS,
            "status": "lkh_timeout",
            "forced_edge_u": task.forced_edge[0],
            "forced_edge_v": task.forced_edge[1],
            "lkh_returncode": None,
            "lkh_wall_seconds": elapsed,
            "safe_threshold_crossed": False,
            "exact_kappa_computed": False,
            "error": f"LKH exceeded {timeout:g} seconds",
        })
    except OSError as exc:
        elapsed = time.perf_counter() - started
        task.stdout_path.write_text("", encoding="utf-8")
        task.stderr_path.write_text(str(exc), encoding="utf-8")
        return finish({
            "schema": RESULT_SCHEMA,
            "semantics": SEMANTICS,
            "status": "lkh_launch_error",
            "forced_edge_u": task.forced_edge[0],
            "forced_edge_v": task.forced_edge[1],
            "lkh_returncode": None,
            "lkh_wall_seconds": elapsed,
            "safe_threshold_crossed": False,
            "exact_kappa_computed": False,
            "error": f"{type(exc).__name__}: {exc}",
        })

    base = {
        "schema": RESULT_SCHEMA,
        "semantics": SEMANTICS,
        "forced_edge_u": task.forced_edge[0],
        "forced_edge_v": task.forced_edge[1],
        "lkh_returncode": int(completed.returncode),
        "lkh_wall_seconds": elapsed,
        "exact_kappa_computed": False,
    }
    if completed.returncode != 0:
        return finish({
            **base,
            "status": "lkh_process_error",
            "safe_threshold_crossed": False,
            "error": "LKH returned a nonzero exit status",
        })
    try:
        tour = _parse_lkh_index_tour(task.tour_path, instance.dimension)
        verified = verify_forced_closure_tour(
            instance,
            baseline_edges,
            task.forced_edge,
            tour,
            baseline_cycle_lower_bound=baseline_cycle_lower_bound,
            baseline_lower_bound_provenance=(
                baseline_lower_bound_provenance
            ),
        )
    except (OSError, ValueError, AssertionError) as exc:
        return finish({
            **base,
            "status": "rejected_unverified_lkh_output",
            "safe_threshold_crossed": False,
            "error": str(exc),
        })
    return finish({**verified, **base})


def _scan_contract(
    instance: TSPLIBInstance,
    baseline_edges: frozenset[Edge],
    candidates: frozenset[Edge],
    *,
    lkh_executable: Path,
    baseline_cycle_lower_bound: float,
    baseline_lower_bound_provenance: str,
    penalty: int,
    runs: int,
    max_trials: int,
    seed: int,
    timeout_seconds: float,
    retain_inputs: bool,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "problem_sha256": instance.source_sha256,
        "dimension": instance.dimension,
        "baseline_edge_count": len(baseline_edges),
        "baseline_edges_sha256": _edge_set_sha256(baseline_edges),
        "candidate_edge_count": len(candidates),
        "candidate_edges_sha256": _edge_set_sha256(candidates),
        "lkh_executable": lkh_executable.name,
        "lkh_executable_sha256": _sha256_file(lkh_executable),
        "declared_baseline_cycle_lower_bound": float(
            baseline_cycle_lower_bound
        ),
        "baseline_lower_bound_provenance": (
            baseline_lower_bound_provenance
        ),
        "penalty": int(penalty),
        "runs": int(runs),
        "max_trials": int(max_trials),
        "master_seed": int(seed),
        "timeout_seconds": float(timeout_seconds),
        "retain_regenerable_lkh_inputs": bool(retain_inputs),
        "semantics": SEMANTICS,
        "ground_truth_inputs": [],
    }


def _result_csv_row(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "forced_edge_u",
        "forced_edge_v",
        "forced_edge_node_u",
        "forced_edge_node_v",
        "status",
        "forced_edge_cost",
        "closure_upper",
        "path_upper",
        "declared_baseline_cycle_lower_bound",
        "safe_gain_lower",
        "safe_kappa_lower",
        "safe_threshold_crossed",
        "lkh_returncode",
        "lkh_wall_seconds",
        "hamiltonicity_verified",
        "forced_edge_verified",
        "baseline_path_membership_verified",
        "objective_recomputed_independently",
        "exact_hamiltonian_path_solved",
        "exact_kappa_computed",
        "resumed",
        "error",
        "task_key",
    )
    return {field: result.get(field, "") for field in fields}


def scan_lkh_forced_closures(
    instance: TSPLIBInstance,
    baseline_edges: Iterable[Edge],
    candidate_edges: Iterable[Edge],
    *,
    lkh_executable: str | Path,
    output_dir: str | Path,
    baseline_cycle_lower_bound: float,
    baseline_lower_bound_provenance: str,
    workers: int = 1,
    penalty: int = 2_000_000,
    runs: int = 1,
    max_trials: int | None = None,
    seed: int = 20260729,
    timeout_seconds: float = 60.0,
    retry_failures: bool = False,
    retain_inputs: bool = False,
) -> dict[str, Any]:
    """Scan nonbaseline candidates with resumable parallel LKH subprocesses."""

    n = instance.dimension
    baseline = _normalize_edge_set(
        baseline_edges,
        n_vertices=n,
        label="baseline_edges",
    )
    candidates = _normalize_edge_set(
        candidate_edges,
        n_vertices=n,
        label="candidate_edges",
    )
    overlap = baseline & candidates
    if overlap:
        raise ValueError(
            "closure candidates must be outside the baseline; "
            f"first overlaps={sorted(overlap)[:5]}"
        )
    executable = Path(lkh_executable)
    if not executable.is_file():
        raise ValueError(f"LKH executable does not exist: {executable}")
    worker_count = _validate_positive_int(workers, label="workers")
    penalty_value = _validate_positive_int(penalty, label="penalty")
    run_count = _validate_positive_int(runs, label="runs")
    trial_count = _validate_positive_int(
        n if max_trials is None else max_trials,
        label="max_trials",
    )
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    lower = float(baseline_cycle_lower_bound)
    if not isfinite(lower) or lower < 0.0:
        raise ValueError(
            "baseline_cycle_lower_bound must be finite and nonnegative"
        )
    provenance = str(baseline_lower_bound_provenance).strip()
    if not provenance:
        raise ValueError("baseline_lower_bound_provenance must not be empty")

    output = Path(output_dir)
    manifest_path = output / "scan_manifest.json"
    contract = _scan_contract(
        instance,
        baseline,
        candidates,
        lkh_executable=executable,
        baseline_cycle_lower_bound=lower,
        baseline_lower_bound_provenance=provenance,
        penalty=penalty_value,
        runs=run_count,
        max_trials=trial_count,
        seed=int(seed),
        timeout_seconds=float(timeout_seconds),
        retain_inputs=retain_inputs,
    )
    contract_hash = _sha256_json(contract)
    manifest = {**contract, "contract_sha256": contract_hash}
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise ValueError(
                "nonempty scan output lacks scan_manifest.json; refusing "
                f"unsafe resume: {output}"
            )
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read existing scan manifest") from exc
        if previous.get("contract_sha256") != contract_hash:
            raise ValueError(
                "existing scan manifest disagrees with the requested contract"
            )
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(manifest_path, manifest)

    edge_root = output / "edges"
    edge_root.mkdir(exist_ok=True)
    sorted_candidates = tuple(sorted(candidates))
    results: dict[Edge, dict[str, Any]] = {}
    resumed_count = 0
    pending: list[tuple[int, Edge, str, Path]] = []

    for candidate_index, candidate in enumerate(sorted_candidates):
        task_seed = int(seed) + candidate_index
        task_dir = edge_root / f"{candidate[0]:06d}_{candidate[1]:06d}"
        result_path = task_dir / "result.json"
        task_key = _sha256_json(
            {
                "contract_sha256": contract_hash,
                "edge": list(candidate),
                "task_seed": task_seed,
            }
        )
        if result_path.is_file():
            try:
                prior = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                prior = None
            if (
                isinstance(prior, dict)
                and prior.get("task_key") == task_key
                and (
                    not retry_failures
                    or prior.get("status") == "verified_closure_witness"
                )
            ):
                restored = dict(prior)
                restored["resumed"] = True
                results[candidate] = restored
                resumed_count += 1
                continue
        pending.append((task_seed, candidate, task_key, task_dir))

    def execute(
        item: tuple[int, Edge, str, Path],
    ) -> tuple[Edge, dict[str, Any]]:
        task_seed, candidate, task_key, task_dir = item
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_lkh_forced_closure(
                instance,
                baseline,
                candidate,
                lkh_executable=executable,
                task_dir=task_dir,
                baseline_cycle_lower_bound=lower,
                baseline_lower_bound_provenance=provenance,
                penalty=penalty_value,
                runs=run_count,
                max_trials=trial_count,
                seed=task_seed,
                timeout_seconds=timeout_seconds,
                retain_inputs=retain_inputs,
            )
        except Exception as exc:  # preserve the remaining 511 independent tasks
            result = {
                "schema": RESULT_SCHEMA,
                "semantics": SEMANTICS,
                "status": "scan_worker_error",
                "forced_edge_u": candidate[0],
                "forced_edge_v": candidate[1],
                "safe_threshold_crossed": False,
                "exact_kappa_computed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result = {
            **result,
            "task_key": task_key,
            "task_seed": task_seed,
            "resumed": False,
        }
        _atomic_json_dump(task_dir / "result.json", result)
        return candidate, result

    if pending:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(execute, item): item[1] for item in pending
            }
            for future in as_completed(future_map):
                candidate, result = future.result()
                results[candidate] = result

    ordered_results = [results[candidate] for candidate in sorted_candidates]
    csv_rows = [_result_csv_row(result) for result in ordered_results]
    csv_path = output / "closure_scan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(csv_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    verified = [
        result
        for result in ordered_results
        if result.get("status") == "verified_closure_witness"
    ]
    crossed = [
        result
        for result in verified
        if bool(result.get("safe_threshold_crossed"))
    ]
    best = (
        max(
            verified,
            key=lambda result: (
                float(result["safe_gain_lower"]),
                -int(result["forced_edge_u"]),
                -int(result["forced_edge_v"]),
            ),
        )
        if verified
        else None
    )
    summary = {
        "schema": SCHEMA,
        "semantics": SEMANTICS,
        "contract_sha256": contract_hash,
        "candidate_count": len(sorted_candidates),
        "executed_this_call_count": len(pending),
        "resumed_count": resumed_count,
        "verified_closure_witness_count": len(verified),
        "safe_threshold_crossing_count": len(crossed),
        "failed_or_unverified_count": len(sorted_candidates) - len(verified),
        "best_safe_gain_lower": (
            float(best["safe_gain_lower"]) if best is not None else None
        ),
        "best_edge": (
            [int(best["forced_edge_u"]), int(best["forced_edge_v"])]
            if best is not None
            else None
        ),
        "exact_kappa_computed": False,
        "regenerable_lkh_problem_and_raw_tour_retained": bool(
            retain_inputs
        ),
        "retained_per_edge_artifacts": [
            "forced_closure.par",
            "lkh.stdout.txt",
            "lkh.stderr.txt",
            "result.json",
        ],
        "interpretation": (
            "positive safe_gain_lower is sufficient conditional on the "
            "declared baseline-cycle lower bound; nonpositive and failed "
            "rows are inconclusive"
        ),
        "closure_scan_csv": csv_path.name,
    }
    _atomic_json_dump(output / "scan_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan LIN318's sealed non-weak candidates with forced-edge LKH "
            "closure witnesses and proof-safe feasible-path lower scores."
        )
    )
    parser.add_argument("--problem", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--lkh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline-cycle-lower-bound",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--baseline-lower-bound-provenance",
        required=True,
    )
    parser.add_argument(
        "--baseline-run-id",
        default="weak_only__r01",
    )
    parser.add_argument(
        "--target-run-id",
        default="static_local_b512__r01",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--penalty", type=int, default=2_000_000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=318)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--retain-lkh-inputs", action="store_true")
    parser.add_argument("--expected-target-edge-count", type=int, default=1500)
    parser.add_argument("--expected-added-edge-count", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    instance = load_euc_2d_instance(args.problem)
    selection = load_candidate_archive(
        args.archive,
        baseline_run_id=args.baseline_run_id,
        target_run_id=args.target_run_id,
        n_vertices=instance.dimension,
        expected_target_edge_count=args.expected_target_edge_count,
        expected_added_edge_count=args.expected_added_edge_count,
    )
    summary = scan_lkh_forced_closures(
        instance,
        selection.baseline_edges,
        selection.added_edges,
        lkh_executable=args.lkh,
        output_dir=args.output,
        baseline_cycle_lower_bound=args.baseline_cycle_lower_bound,
        baseline_lower_bound_provenance=(
            args.baseline_lower_bound_provenance
        ),
        workers=args.workers,
        penalty=args.penalty,
        runs=args.runs,
        max_trials=args.max_trials,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
        retry_failures=args.retry_failures,
        retain_inputs=args.retain_lkh_inputs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = [
    "CandidateArchiveSelection",
    "LKHClosureTask",
    "SEMANTICS",
    "load_candidate_archive",
    "normalize_edge",
    "prepare_lkh_closure_task",
    "run_lkh_forced_closure",
    "scan_lkh_forced_closures",
    "verify_forced_closure_tour",
]


if __name__ == "__main__":
    main()
