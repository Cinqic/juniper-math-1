"""Tests for the dependency/license cross-check.

The Phase 0 review candidate declared NumPy as a runtime dependency but had
no license entry for it, and `manifests-validate` passed anyway because
nothing compared pyproject.toml against manifests/licenses.yaml. These tests
make that omission impossible to reintroduce silently.
See reports/OPUS5_PHASE0_REVIEW.md (F-05).
"""

from __future__ import annotations

import textwrap

import pytest

from juniper_math.errors import JuniperManifestError
from juniper_math.manifests import (
    check_dependency_licenses,
    declared_dependencies,
    load_licenses_manifest,
    normalize_package_name,
)


def test_all_declared_dependencies_are_licensed():
    results = check_dependency_licenses()
    failures = [(pkg, detail) for pkg, ok, detail in results if not ok]
    assert failures == [], f"unlicensed or mismatched dependencies: {failures}"


def test_numpy_specifically_is_licensed():
    """The exact gap found in the Phase 0 review candidate."""
    entries = {e.get("package"): e for e in load_licenses_manifest()}
    assert "numpy" in entries, "NumPy is a declared runtime dependency and must be licensed"
    assert entries["numpy"]["scope"] == "dependency"
    assert entries["numpy"]["spdx_identifier"] not in ("", None, "UNKNOWN")


def test_every_runtime_dependency_has_an_entry():
    declared = declared_dependencies()
    licensed = {
        normalize_package_name(str(e["package"])) for e in load_licenses_manifest() if e.get("package")
    }
    assert declared["runtime"] <= licensed
    assert declared["dev"] <= licensed
    assert declared["runtime"] == {"torch", "pyyaml", "numpy", "sentencepiece"}


def test_missing_license_entry_is_detected(tmp_path):
    """Remove NumPy from a copy of the manifest; the check must fail."""
    entries = [e for e in load_licenses_manifest() if e.get("package") != "numpy"]
    manifest = tmp_path / "licenses.yaml"
    import yaml

    manifest.write_text(yaml.safe_dump({"licenses": entries}), encoding="utf-8")

    results = check_dependency_licenses(licenses_path=manifest)
    failures = {pkg for pkg, ok, _ in results if not ok}
    assert "numpy" in failures


def test_new_undeclared_dependency_is_detected(tmp_path):
    """Adding a dependency without licensing it must fail the cross-check."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent("""
            [project]
            name = "juniper-math-1"
            dependencies = ["torch", "pyyaml", "numpy", "requests>=2.0"]
            """).strip(),
        encoding="utf-8",
    )
    results = check_dependency_licenses(pyproject_path=pyproject)
    failures = {pkg: detail for pkg, ok, detail in results if not ok}
    assert "requests" in failures
    assert "no entry" in failures["requests"]


def test_stale_license_entry_is_detected(tmp_path):
    """A license entry for a package nothing declares is flagged."""
    entries = list(load_licenses_manifest())
    entries.append(
        {
            "license_id": "ghost",
            "scope": "dependency",
            "package": "not-a-real-dependency",
            "spdx_identifier": "MIT",
            "reference": "n/a",
            "attribution_required": True,
            "restrictions": "none",
            "redistribution_status": "ALLOWED",
            "notes": "test fixture",
        }
    )
    manifest = tmp_path / "licenses.yaml"
    import yaml

    manifest.write_text(yaml.safe_dump({"licenses": entries}), encoding="utf-8")
    results = check_dependency_licenses(licenses_path=manifest)
    failures = {pkg: detail for pkg, ok, detail in results if not ok}
    assert "not-a-real-dependency" in failures
    assert "stale" in failures["not-a-real-dependency"]


def test_dependency_scoped_entry_must_declare_a_package(tmp_path):
    entries = list(load_licenses_manifest())
    entries.append(
        {
            "license_id": "no-package-field",
            "scope": "dependency",
            "spdx_identifier": "MIT",
            "reference": "n/a",
            "attribution_required": True,
            "restrictions": "none",
            "redistribution_status": "ALLOWED",
            "notes": "test fixture",
        }
    )
    manifest = tmp_path / "licenses.yaml"
    import yaml

    manifest.write_text(yaml.safe_dump({"licenses": entries}), encoding="utf-8")
    with pytest.raises(JuniperManifestError, match="must declare a 'package' field"):
        load_licenses_manifest(manifest)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("types-PyYAML", "types-pyyaml"), ("types_PyYAML", "types-pyyaml"), ("PyYAML", "pyyaml")],
)
def test_package_name_normalization(raw, expected):
    assert normalize_package_name(raw) == expected


def test_requirement_specifiers_are_stripped():
    """Version specifiers and extras must not leak into dependency names."""
    declared = declared_dependencies()
    for name in declared["runtime"] | declared["dev"]:
        assert not any(ch in name for ch in "<>=!~[; ")
