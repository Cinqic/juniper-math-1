#!/usr/bin/env bash
# Safe repository-level bootstrap for Linux.
#
# This script sets up a Python virtual environment and installs project
# dependencies. It does NOT touch system packages, GPU drivers, boot
# configuration, or anything outside this repository's virtual environment.
# System-level prerequisites (NVIDIA driver, Python interpreter itself) are
# documented in docs/ENVIRONMENT.md and docs/RECOVERY.md — install those
# yourself first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "FAIL: $PYTHON_BIN not found on PATH. Install Python 3.10-3.12 first." >&2
    exit 1
fi

echo "Using $($PYTHON_BIN --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv ..."
    "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip >/dev/null
echo "Installing project dependencies (editable, with dev extras) ..."
pip install -e ".[dev]"

echo
echo "Running environment validation ..."
python -m juniper_math validate-env || true

echo
echo "Running test suite ..."
pytest -q

echo
echo "Bootstrap complete. Activate with: source .venv/bin/activate"
