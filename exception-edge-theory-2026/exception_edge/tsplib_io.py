from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from math import floor, isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class TSPLIBError(ValueError):
    """Base class for malformed or inconsistent TSPLIB artifacts."""


class TSPLIBParseError(TSPLIBError):
    """Raised when a TSPLIB file cannot be parsed."""


class TSPLIBValidationError(TSPLIBError):
    """Raised when individually valid artifacts disagree."""


@dataclass(frozen=True)
class TSPLIBInstance:
    name: str
    dimension: int
    edge_weight_type: str
    node_ids: tuple[int, ...]
    points: np.ndarray
    distances: np.ndarray
    headers: Mapping[str, str]
    source_path: str
    source_sha256: str

    @property
    def node_to_index(self) -> dict[int, int]:
        return {
            node_id: index for index, node_id in enumerate(self.node_ids)
        }


@dataclass(frozen=True)
class TSPLIBTour:
    name: str
    tour: tuple[int, ...]
    original_node_ids: tuple[int, ...]
    cost: int
    headers: Mapping[str, str]
    source_path: str
    source_sha256: str


def tsplib_nint(value: float) -> int:
    """TSPLIB's ``nint`` rule: round a nonnegative value via floor(x + 0.5)."""
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise TSPLIBValidationError(
            f"TSPLIB nint requires a finite nonnegative value, got {value!r}"
        )
    return int(floor(value + 0.5))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_lines(path_like: str | Path) -> tuple[Path, list[str]]:
    path = Path(path_like)
    if not path.is_file():
        raise TSPLIBParseError(f"TSPLIB file does not exist: {path}")
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
                return path, handle.read().splitlines()
        return path, path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TSPLIBParseError(f"cannot read TSPLIB file {path}: {exc}") from exc


def _header_entry(line: str, path: Path, line_number: int) -> tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
    else:
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise TSPLIBParseError(
                f"{path}:{line_number}: malformed header line {line!r}"
            )
        key, value = fields
    key = key.strip().upper()
    value = value.strip()
    if not key or not value:
        raise TSPLIBParseError(
            f"{path}:{line_number}: malformed header line {line!r}"
        )
    return key, value


def _required_header(
    headers: Mapping[str, str], key: str, path: Path
) -> str:
    value = headers.get(key)
    if value is None:
        raise TSPLIBParseError(f"{path}: missing required header {key}")
    return value


def _parse_positive_dimension(headers: Mapping[str, str], path: Path) -> int:
    raw = _required_header(headers, "DIMENSION", path)
    try:
        dimension = int(raw)
    except ValueError as exc:
        raise TSPLIBParseError(
            f"{path}: DIMENSION must be an integer, got {raw!r}"
        ) from exc
    if dimension < 3:
        raise TSPLIBValidationError(
            f"{path}: DIMENSION must be at least 3, got {dimension}"
        )
    return dimension


def _euc_2d_distances(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    euclidean = np.sqrt(np.sum(delta * delta, axis=2))
    distances = np.floor(euclidean + 0.5).astype(np.int64)
    np.fill_diagonal(distances, 0)
    distances.setflags(write=False)
    return distances


def load_euc_2d_instance(path_like: str | Path) -> TSPLIBInstance:
    """Parse a TSPLIB ``EUC_2D`` TSP from ``.tsp`` or ``.tsp.gz``."""
    path, lines = _read_lines(path_like)
    headers: dict[str, str] = {}
    coordinates: list[tuple[int, float, float]] = []
    in_coordinates = False
    found_section = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper == "NODE_COORD_SECTION":
            if found_section:
                raise TSPLIBParseError(
                    f"{path}:{line_number}: duplicate NODE_COORD_SECTION"
                )
            found_section = True
            in_coordinates = True
            continue
        if in_coordinates:
            if upper.endswith("_SECTION"):
                raise TSPLIBParseError(
                    f"{path}:{line_number}: unexpected section {line!r} "
                    "inside NODE_COORD_SECTION"
                )
            fields = line.split()
            if len(fields) != 3:
                raise TSPLIBParseError(
                    f"{path}:{line_number}: expected 'node_id x y', got {line!r}"
                )
            try:
                node_id = int(fields[0])
                x = float(fields[1])
                y = float(fields[2])
            except ValueError as exc:
                raise TSPLIBParseError(
                    f"{path}:{line_number}: invalid coordinate row {line!r}"
                ) from exc
            if not isfinite(x) or not isfinite(y):
                raise TSPLIBValidationError(
                    f"{path}:{line_number}: coordinates must be finite"
                )
            coordinates.append((node_id, x, y))
            continue
        key, value = _header_entry(line, path, line_number)
        headers[key] = value

    if not found_section:
        raise TSPLIBParseError(f"{path}: missing NODE_COORD_SECTION")
    instance_type = _required_header(headers, "TYPE", path).upper()
    if instance_type != "TSP":
        raise TSPLIBValidationError(
            f"{path}: TYPE must be TSP, got {instance_type!r}"
        )
    edge_weight_type = _required_header(
        headers, "EDGE_WEIGHT_TYPE", path
    ).upper()
    if edge_weight_type != "EUC_2D":
        raise TSPLIBValidationError(
            f"{path}: only EDGE_WEIGHT_TYPE EUC_2D is supported, "
            f"got {edge_weight_type!r}"
        )
    dimension = _parse_positive_dimension(headers, path)
    if len(coordinates) != dimension:
        raise TSPLIBValidationError(
            f"{path}: DIMENSION is {dimension}, but NODE_COORD_SECTION "
            f"contains {len(coordinates)} rows"
        )
    node_ids = tuple(row[0] for row in coordinates)
    if len(set(node_ids)) != dimension:
        duplicates = sorted(
            node_id for node_id in set(node_ids) if node_ids.count(node_id) > 1
        )
        raise TSPLIBValidationError(
            f"{path}: duplicate node identifiers: {duplicates}"
        )
    points = np.asarray(
        [(row[1], row[2]) for row in coordinates],
        dtype=float,
    )
    points.setflags(write=False)
    return TSPLIBInstance(
        name=headers.get("NAME", path.name),
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        node_ids=node_ids,
        points=points,
        distances=_euc_2d_distances(points),
        headers=dict(headers),
        source_path=str(path.resolve()),
        source_sha256=_file_sha256(path),
    )


def validate_tour(
    instance: TSPLIBInstance, tour: Sequence[int]
) -> tuple[int, ...]:
    """Validate and canonicalize an already 0-based Hamiltonian tour."""
    normalized = tuple(int(vertex) for vertex in tour)
    if len(normalized) != instance.dimension:
        raise TSPLIBValidationError(
            f"tour has {len(normalized)} vertices; expected {instance.dimension}"
        )
    expected = set(range(instance.dimension))
    actual = set(normalized)
    if len(actual) != len(normalized):
        raise TSPLIBValidationError("tour contains duplicate vertices")
    missing = sorted(expected - actual)
    outside = sorted(actual - expected)
    if missing or outside:
        raise TSPLIBValidationError(
            f"tour is not Hamiltonian; missing={missing}, outside={outside}"
        )
    return normalized


def tour_cost(instance: TSPLIBInstance, tour: Sequence[int]) -> int:
    """Return the closed-cycle cost under the parsed TSPLIB integer metric."""
    normalized = validate_tour(instance, tour)
    return int(
        sum(
            int(
                instance.distances[
                    normalized[index],
                    normalized[(index + 1) % len(normalized)],
                ]
            )
            for index in range(len(normalized))
        )
    )


def load_tsplib_tour(
    path_like: str | Path,
    instance: TSPLIBInstance,
) -> TSPLIBTour:
    """Parse TOUR/``.opt.tour`` data and map its node IDs to 0-based indices."""
    path, lines = _read_lines(path_like)
    headers: dict[str, str] = {}
    original_ids: list[int] = []
    in_tour = False
    found_section = False
    terminated = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper == "TOUR_SECTION":
            if found_section:
                raise TSPLIBParseError(
                    f"{path}:{line_number}: duplicate TOUR_SECTION"
                )
            found_section = True
            in_tour = True
            continue
        if in_tour:
            for token in line.split():
                try:
                    node_id = int(token)
                except ValueError as exc:
                    raise TSPLIBParseError(
                        f"{path}:{line_number}: invalid TOUR_SECTION "
                        f"token {token!r}"
                    ) from exc
                if node_id == -1:
                    terminated = True
                    in_tour = False
                    break
                if terminated:
                    raise TSPLIBParseError(
                        f"{path}:{line_number}: data follows TOUR_SECTION -1"
                    )
                original_ids.append(node_id)
            continue
        key, value = _header_entry(line, path, line_number)
        headers[key] = value

    if not found_section:
        raise TSPLIBParseError(f"{path}: missing TOUR_SECTION")
    if not terminated:
        raise TSPLIBParseError(f"{path}: TOUR_SECTION is missing -1 terminator")
    tour_type = headers.get("TYPE", "TOUR").upper()
    if tour_type != "TOUR":
        raise TSPLIBValidationError(
            f"{path}: TYPE must be TOUR, got {tour_type!r}"
        )
    if "DIMENSION" in headers:
        tour_dimension = _parse_positive_dimension(headers, path)
        if tour_dimension != instance.dimension:
            raise TSPLIBValidationError(
                f"{path}: tour DIMENSION {tour_dimension} disagrees with "
                f"instance DIMENSION {instance.dimension}"
            )
    node_to_index = instance.node_to_index
    unknown = sorted(set(original_ids) - set(node_to_index))
    if unknown:
        raise TSPLIBValidationError(
            f"{path}: tour contains node IDs absent from the instance: {unknown}"
        )
    mapped = tuple(node_to_index[node_id] for node_id in original_ids)
    mapped = validate_tour(instance, mapped)
    return TSPLIBTour(
        name=headers.get("NAME", path.name),
        tour=mapped,
        original_node_ids=tuple(original_ids),
        cost=tour_cost(instance, mapped),
        headers=dict(headers),
        source_path=str(path.resolve()),
        source_sha256=_file_sha256(path),
    )


def build_known_optimum_manifest(
    instance: TSPLIBInstance,
    tour: TSPLIBTour,
    *,
    expected_optimum_cost: int | None = None,
) -> dict[str, Any]:
    """Build a portable, hash-backed manifest and verify an optional optimum."""
    if expected_optimum_cost is not None:
        expected_optimum_cost = int(expected_optimum_cost)
        if tour.cost != expected_optimum_cost:
            raise TSPLIBValidationError(
                f"tour cost {tour.cost} disagrees with expected optimum "
                f"{expected_optimum_cost}"
            )
    return {
        "schema": "tsplib-known-optimum-v1",
        "instance_name": instance.name,
        "dimension": instance.dimension,
        "edge_weight_type": instance.edge_weight_type,
        "node_ids": list(instance.node_ids),
        "instance_file_sha256": instance.source_sha256,
        "tour_name": tour.name,
        "tour_file_sha256": tour.source_sha256,
        "tour_cost": tour.cost,
        "expected_optimum_cost": expected_optimum_cost,
    }


def validate_known_optimum_manifest(
    manifest: Mapping[str, Any],
    instance: TSPLIBInstance,
    tour: TSPLIBTour,
) -> None:
    """Raise a clear error if a manifest does not describe these artifacts."""
    expected = build_known_optimum_manifest(
        instance,
        tour,
        expected_optimum_cost=(
            int(manifest["expected_optimum_cost"])
            if manifest.get("expected_optimum_cost") is not None
            else None
        ),
    )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise TSPLIBValidationError(
                f"manifest field {key!r} is {manifest.get(key)!r}; "
                f"expected {value!r}"
            )


def manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest deterministically for hashing or storage."""
    return json.dumps(
        dict(manifest),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
