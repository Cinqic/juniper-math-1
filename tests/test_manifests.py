from __future__ import annotations

import pytest
import yaml

from juniper_math.errors import JuniperManifestError
from juniper_math.manifests import (
    load_artifacts_manifest,
    load_licenses_manifest,
    load_sources_manifest,
    verify_artifacts_manifest,
)


def test_sources_manifest_loads():
    entries = load_sources_manifest()
    assert isinstance(entries, list)


def test_licenses_manifest_loads_and_has_project_license():
    entries = load_licenses_manifest()
    ids = {entry["license_id"] for entry in entries}
    assert "juniper-math-1-code" in ids


def test_artifacts_manifest_loads_and_has_entries():
    entries = load_artifacts_manifest()
    assert len(entries) > 0


def test_artifacts_manifest_hashes_verify():
    results = verify_artifacts_manifest()
    assert len(results) > 0
    for artifact_id, ok, detail in results:
        assert ok, f"{artifact_id}: {detail}"


def test_corrupted_hash_detected(tmp_path, monkeypatch):
    # verify_artifacts_manifest resolves entry["path"] relative to REPO_ROOT;
    # point REPO_ROOT at a scratch directory containing a known file and a
    # manifest with a deliberately wrong recorded hash.
    (tmp_path / "sample.txt").write_text("frozen content", encoding="utf-8")
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "artifacts": [
                    {
                        "artifact_id": "fake",
                        "path": "sample.txt",
                        "sha256": "0" * 64,
                        "description": "deliberately wrong hash",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("juniper_math.manifests.REPO_ROOT", tmp_path)
    results = verify_artifacts_manifest(manifest)
    assert results == [("fake", False, results[0][2])]
    assert "mismatch" in results[0][2]


def test_missing_artifact_file_detected(tmp_path, monkeypatch):
    manifest = tmp_path / "artifacts.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "artifacts": [
                    {
                        "artifact_id": "ghost",
                        "path": "does_not_exist.txt",
                        "sha256": "0" * 64,
                        "description": "file was never created",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("juniper_math.manifests.REPO_ROOT", tmp_path)
    results = verify_artifacts_manifest(manifest)
    assert results[0][1] is False
    assert "missing file" in results[0][2]


def test_duplicate_source_id_rejected(tmp_path):
    bad = tmp_path / "sources.yaml"
    entry = {
        "source_id": "dup",
        "source_name": "x",
        "publisher": "x",
        "source_url": "https://example.invalid",
        "source_version": "1",
        "acquisition_date": "2026-01-01",
        "intended_use": "test",
        "license_id": "UNKNOWN",
        "redistribution_status": "REQUIRES_REVIEW",
        "transformation_status": "none",
        "checksum": "UNKNOWN",
        "notes": "",
    }
    bad.write_text(yaml.safe_dump({"sources": [entry, dict(entry)]}), encoding="utf-8")
    with pytest.raises(JuniperManifestError, match="duplicate source_id"):
        load_sources_manifest(bad)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(JuniperManifestError, match="not found"):
        load_sources_manifest(tmp_path / "missing.yaml")
