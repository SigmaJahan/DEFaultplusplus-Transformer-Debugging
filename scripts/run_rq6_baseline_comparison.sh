#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/3_Comparison_with_defaultplusplus"
python run_rq6_baselines.py "$@"
