#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/diagnosis_root_cause/ndg_diagnosis_builder"
python run_stage3_diagnosis_ndg_v1.py "$@"
