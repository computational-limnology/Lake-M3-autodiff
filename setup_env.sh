#!/usr/bin/env bash
# Create a local virtual environment with the packages needed to run the
# scripts in src/ (numpy/pandas reference model + JAX autodiff port).
#
# A virtual environment is a folder of platform-specific binaries -- it
# cannot be copied between computers (or between operating systems) and
# expected to work. Run this script ON EACH computer you want to test on;
# it takes about a minute and only needs requirements.txt alongside it.
#
# Usage:
#   ./setup_env.sh
#   source .venv/bin/activate
#   python src/calibrate_M3_jax.py Ravn --sensitivity-only
#
# Windows (PowerShell), equivalent steps:
#   py -m venv .venv
#   .venv\Scripts\Activate.ps1
#   pip install -r requirements.txt

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "Creating .venv with $($PYTHON --version)..."
$PYTHON -m venv .venv

echo "Installing pinned dependencies from requirements.txt..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then, e.g.:            python src/calibrate_M3_jax.py Ravn --sensitivity-only"
