#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/2_Diagnosis_Root_Cause/ndg_diagnosis_builder"
python run_stage3_diagnosis_ndg_v1.py "$@"
