#!/usr/bin/env bash
# Safe repository-level bootstrap for Linux.
#
# This script sets up a Python virtual environment and installs project
# dependencies. It does NOT touch system packages, GPU drivers, boot
# configuration, or anything outside this repository's virtual environment.
# It never invokes sudo: when a system prerequisite is missing it tells you
# what to install and stops. System-level prerequisites (NVIDIA driver, the
# Python interpreter, python3-venv) are documented in docs/ENVIRONMENT.md and
# docs/RECOVERY.md — install those yourself first.
#
# By default this installs the EXACT validated Phase 0 environment from
# requirements-lock.txt. Set JUNIPER_USE_RANGES=1 to resolve from
# pyproject.toml's compatibility ranges instead (useful when deliberately
# testing a dependency upgrade).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "FAIL: $PYTHON_BIN not found on PATH. Install Python 3.12 first." >&2
    echo "      On Debian/Ubuntu/Linux Mint: sudo apt install python3 python3-venv python3-pip" >&2
    exit 1
fi

echo "Using $($PYTHON_BIN --version)"

# --- Preflight: venv support -------------------------------------------------
# On Debian-family distributions (Debian, Ubuntu, Linux Mint) the standard
# library's venv/ensurepip support ships in a SEPARATE package. Installing
# "python3" alone leaves `python3 -m venv` broken with an ensurepip error that
# is confusing if you hit it mid-recovery. Detect it up front and say exactly
# what to install. This script deliberately does NOT run apt itself.
if ! "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
    PY_MM="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    cat >&2 <<EOF
FAIL: this Python cannot create virtual environments — the 'ensurepip' module
      is missing, so '$PYTHON_BIN -m venv' will fail.

      This is normal on Debian, Ubuntu, and Linux Mint: venv support is
      packaged separately from the interpreter. Install it, then re-run:

          sudo apt update
          sudo apt install python3-venv python3-pip

      (If that package name is not found, try the version-specific one:
          sudo apt install python${PY_MM}-venv
      )

      This script will not install system packages for you.
EOF
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv ..."
    "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip >/dev/null

if [ "${JUNIPER_USE_RANGES:-0}" = "1" ]; then
    echo "Installing from pyproject.toml compatibility ranges (JUNIPER_USE_RANGES=1) ..."
    pip install -e ".[dev]"
else
    echo "Installing the exact validated environment from requirements-lock.txt ..."
    pip install -r requirements-lock.txt
    pip install -e . --no-deps
fi

echo
echo "Running environment validation ..."
python -m juniper_math validate-env || true

echo
echo "Running test suite ..."
pytest -q

echo
echo "Bootstrap complete. Activate with: source .venv/bin/activate"
