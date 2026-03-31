#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing package with dev dependencies..."
pip install -e ".[dev]"

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
