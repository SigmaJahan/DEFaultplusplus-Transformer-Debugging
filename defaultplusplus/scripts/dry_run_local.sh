#!/usr/bin/env bash
# End-to-end local dry run for DEFault++.
#
# Exercises every module at minimum scale before any Compute Canada
# submission. Catches:
#   - import / type-signature drift,
#   - regression in the FPG or feature-group code,
#   - injector context-manager mistakes,
#   - feature-construction shape errors,
#   - training-loop wiring problems with the new loss / kwarg names.
#
# Usage:
#   bash scripts/dry_run_local.sh            # full smoke test
#   bash scripts/dry_run_local.sh --quick    # imports + pytest dry-run only
#   bash scripts/dry_run_local.sh --train    # also do a 1-epoch training pass

set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "${CODE_ROOT}"

QUICK=0
WITH_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --train) WITH_TRAIN=1 ;;
    *) echo "[dry-run] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "=================================================================="
echo "  DEFault++ local dry run"
echo "  CODE_ROOT = ${CODE_ROOT}"
echo "=================================================================="

# 1. Import smoke test ---------------------------------------------------
echo ""
echo "[1/4] Import smoke test"
python - <<'PY'
import importlib, sys

modules = [
    # Data + FPG
    "src.data.feature_groups",
    "src.data.feature_processor",
    "src.data.fundamental_fpg",
    "src.data.loader",
    # Models
    "src.models.group_encoder",
    # DEForm + benchmark
    "src.defaultplusplus.deform",
    "src.defaultplusplus.deform.operators",
    "src.defaultplusplus.deform.fault_config",
    "src.defaultplusplus.deform.injection",
    "src.defaultplusplus.deform.validation",
    "src.defaultplusplus.benchmark",
    "src.defaultplusplus.benchmark.config_grid",
    "src.defaultplusplus.benchmark.runner",
    "src.defaultplusplus.benchmark.dataset_writer",
    # Public package + extraction
    "src.defaultplusplus.extraction",
    "src.defaultplusplus.extraction.feature_construction",
    "src.defaultplusplus.api",
    # Training pipeline
    "hierarchical_graph_category_rootcause.model",
    "hierarchical_graph_category_rootcause.losses",
]

failed = []
for m in modules:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        failed.append((m, e))

if failed:
    print(f"\n{len(failed)} module(s) failed to import.")
    sys.exit(1)
print("\nAll imports succeeded.")
PY

# 2. Pytest smoke suite --------------------------------------------------
echo ""
echo "[2/4] Pytest smoke suite (tests/test_dry_run.py)"
python -m pytest tests/test_dry_run.py -q --maxfail=1

# 3. FPG / feature-group sanity ------------------------------------------
echo ""
echo "[3/4] FPG + feature-group sanity"
python - <<'PY'
from src.data.feature_groups import (
    SUBSYSTEM_GROUPS, STRUCTURAL_GROUPS, NON_STRUCTURAL_GROUPS,
    build_group_indices,
)
from src.data.fundamental_fpg import (
    fundamental_to_feature_group_adjacency,
    build_fundamental_fpg,
)

# Encoder & decoder both build without error.
for arch in ("encoder", "decoder"):
    names, adj, meta = fundamental_to_feature_group_adjacency(arch)
    assert adj.shape == (len(names), len(names))
    print(f"  {arch}: groups={names}")
    print(f"  {arch}: adj shape={adj.shape}, "
          f"row sums (raw) min={float(adj.sum(axis=1).min()):.2f}, "
          f"max={float(adj.sum(axis=1).max()):.2f}")

# Group routing handles a representative feature-name list.
sample_names = [
    "attn_entropy", "attn_pad_mass", "qk_cos", "score_mean",
    "ffn_norm", "ln_gamma", "res_cos", "cka_l3_l4",
    "emb_norm", "logit_conf", "accuracy", "loss",
    "step_time", "grad_norm_attn",
]
gi = build_group_indices(sample_names)
print(f"  group_indices keys: {sorted(gi)}")
assert "attention" in gi
assert "training_dynamics" in gi
print("FPG + feature-group sanity: OK")
PY

if [[ ${QUICK} -eq 1 && ${WITH_TRAIN} -eq 0 ]]; then
  echo ""
  echo "Dry run complete (--quick). Skipping the 1-epoch training pass."
  exit 0
fi

# 4. Optional 1-epoch training pass --------------------------------------
if [[ ${WITH_TRAIN} -eq 1 ]]; then
  echo ""
  echo "[4/4] 1-epoch training pass on encoder (smoke)"
  if [[ ! -f "../data/encoder_dataset.csv" ]]; then
    echo "  Skipping: DEFault-bench CSV not found (run benchmark construction first)."
    exit 0
  fi
  python -m hierarchical_graph_category_rootcause.train \
      --arch encoder --epochs 1 \
      --output "../results/dry_run"
fi

echo ""
echo "Dry run complete."
