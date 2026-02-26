#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$ROOT/results"
DIAG="$RESULTS/diagnosis"
OUT="$DIAG"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  # Optional local venv for reproducibility.
  # Script also works with the active system/conda environment.
  source "$ROOT/.venv/bin/activate"
fi

PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python -m ndg_graph.cli \
  --enc_detection     "$RESULTS/detection/enc_detection.json" \
  --enc_categorization "$RESULTS/categorization/enc_categorization.json" \
  --xai_enc           "$RESULTS/explanations/xai_enc_categorization.json" \
  --dec_detection     "$RESULTS/detection/dec_detection.json" \
  --dec_categorization "$RESULTS/categorization/dec_categorization.json" \
  --xai_dec           "$RESULTS/explanations/xai_dec_categorization.json" \
  --feature_core_map  "$SCRIPT_DIR/ndg_graph/feature_core_map.md" \
  --enc_diagnosis     "$DIAG/enc_diagnosis.json" \
  --dec_diagnosis     "$DIAG/dec_diagnosis.json" \
  --out_dir           "$OUT" \
  --plots

echo "Done. Output in $OUT"
