#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src/diagnosis_root_cause/ndg_diagnosis_builder"
python build_diagnosis_ndg.py "$@"
