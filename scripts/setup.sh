#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
  echo "Creating virtual environment in $ROOT/.venv"
  python3 -m venv --system-site-packages .venv
else
  echo "Reusing existing virtual environment in $ROOT/.venv"
fi

source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e ".[dev]"
