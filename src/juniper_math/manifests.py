"""Provenance manifest loading and validation: sources, licenses, artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperManifestError
from juniper_math.hashing import sha256_file
from juniper_math.paths import MANIFESTS_DIR, REPO_ROOT

SOURCES_PATH = MANIFESTS_DIR / "sources.yaml"
LICENSES_PATH = MANIFESTS_DIR / "licenses.yaml"
ARTIFACTS_PATH = MANIFESTS_DIR / "artifacts.yaml"

_SOURCE_FIELDS = {
    "source_id",
    "source_name",
    "publisher",
    "source_url",
    "source_version",
    "acquisition_date",
    "intended_use",
    "license_id",
    "redistribution_status",
    "transformation_status",
    "checksum",
    "notes",
}

_LICENSE_FIELDS = {
    "license_id",
    "scope",
    "spdx_identifier",
    "reference",
    "attribution_required",
    "restrictions",
    "redistribution_status",
    "notes",
}

_ARTIFACT_FIELDS = {"artifact_id", "path", "sha256", "description"}

_VALID_REDISTRIBUTION = {"ALLOWED", "NOT_ALLOWED", "REQUIRES_REVIEW", "NOT_APPLICABLE"}


def _load_yaml_list(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise JuniperManifestError(f"Manifest not found at {path}.")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperManifestError(f"{path}: invalid YAML ({exc}).") from exc
    if not isinstance(raw, dict) or key not in raw or not isinstance(raw[key], list):
        raise JuniperManifestError(f"{path}: expected top-level mapping with list field '{key}'.")
    return raw[key]


def load_sources_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or SOURCES_PATH
    entries = _load_yaml_list(source, "sources")
    seen_ids: set[str] = set()
    for entry in entries:
        missing = _SOURCE_FIELDS - entry.keys()
        if missing:
            raise JuniperManifestError(f"{source}: source entry missing field(s): {sorted(missing)}")
        if entry["source_id"] in seen_ids:
            raise JuniperManifestError(f"{source}: duplicate source_id {entry['source_id']!r}")
        seen_ids.add(entry["source_id"])
        if entry["redistribution_status"] not in _VALID_REDISTRIBUTION:
            raise JuniperManifestError(
                f"{source}: source {entry['source_id']!r} has invalid redistribution_status"
            )
    return entries


def load_licenses_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or LICENSES_PATH
    entries = _load_yaml_list(source, "licenses")
    seen_ids: set[str] = set()
    for entry in entries:
        missing = _LICENSE_FIELDS - entry.keys()
        if missing:
            raise JuniperManifestError(f"{source}: license entry missing field(s): {sorted(missing)}")
        if entry["license_id"] in seen_ids:
            raise JuniperManifestError(f"{source}: duplicate license_id {entry['license_id']!r}")
        seen_ids.add(entry["license_id"])
        if not isinstance(entry["attribution_required"], bool):
            raise JuniperManifestError(
                f"{source}: license {entry['license_id']!r} 'attribution_required' must be bool"
            )
    return entries


def load_artifacts_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or ARTIFACTS_PATH
    entries = _load_yaml_list(source, "artifacts")
    seen_ids: set[str] = set()
    for entry in entries:
        missing = _ARTIFACT_FIELDS - entry.keys()
        if missing:
            raise JuniperManifestError(f"{source}: artifact entry missing field(s): {sorted(missing)}")
        if entry["artifact_id"] in seen_ids:
            raise JuniperManifestError(f"{source}: duplicate artifact_id {entry['artifact_id']!r}")
        seen_ids.add(entry["artifact_id"])
    return entries


def verify_artifacts_manifest(path: Path | None = None) -> list[tuple[str, bool, str]]:
    """Returns (artifact_id, ok, detail) for every artifact in the manifest."""
    entries = load_artifacts_manifest(path)
    results = []
    for entry in entries:
        artifact_path = REPO_ROOT / entry["path"]
        if not artifact_path.is_file():
            results.append((entry["artifact_id"], False, f"missing file: {artifact_path}"))
            continue
        actual = sha256_file(artifact_path)
        expected = entry["sha256"].lower()
        if actual == expected:
            results.append((entry["artifact_id"], True, "sha256 match"))
        else:
            results.append(
                (entry["artifact_id"], False, f"sha256 mismatch: expected {expected}, got {actual}")
            )
    return results
