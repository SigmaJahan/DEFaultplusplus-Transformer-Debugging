#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/2_Diagnosis_Root_Cause"
python run_rq5_signature_matching.py "$@"
