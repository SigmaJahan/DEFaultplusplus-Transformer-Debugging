#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$ROOT/results"
DIAG="$RESULTS/stage_3_diagnosis"
OUT="$DIAG"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  # Optional local venv for reproducibility.
  # Script also works with the active system/conda environment.
  source "$ROOT/.venv/bin/activate"
fi

PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python -m ndg_stage3.cli \
  --enc_detection     "$RESULTS/stage_1_detection/enc_detection.json" \
  --enc_categorization "$RESULTS/stage_2_categorization/enc_categorization.json" \
  --xai_enc           "$RESULTS/stage_2_1_categorization_xai/xai_enc_categorization.json" \
  --dec_detection     "$RESULTS/stage_1_detection/dec_detection.json" \
  --dec_categorization "$RESULTS/stage_2_categorization/dec_categorization.json" \
  --xai_dec           "$RESULTS/stage_2_1_categorization_xai/xai_dec_categorization.json" \
  --feature_core_map  "$SCRIPT_DIR/ndg_stage3/feature_core_map.md" \
  --enc_diagnosis     "$DIAG/enc_diagnosis.json" \
  --dec_diagnosis     "$DIAG/dec_diagnosis.json" \
  --out_dir           "$OUT" \
  --plots

echo "Done. Output in $OUT"
