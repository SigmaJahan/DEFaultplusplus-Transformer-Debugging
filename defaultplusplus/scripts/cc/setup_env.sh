#!/usr/bin/env bash
# One-shot virtual-environment build for Compute Canada.
#
# Run this once from a login node:
#
#     bash scripts/cc/setup_env.sh
#
# The venv is created in $SCRATCH/venvs/defaultpp because compute nodes
# cannot read $HOME on most CC clusters. After this script completes,
# every SLURM job activates the same venv via env.sh.

set -euo pipefail

module --quiet purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

: "${SCRATCH:?SCRATCH is not set; run on a CC login node}"

CODE_ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
VENV_ROOT="${SCRATCH}/venvs/defaultpp"

if [[ -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "[setup_env] venv already exists at ${VENV_ROOT}; reusing"
else
  echo "[setup_env] creating venv at ${VENV_ROOT}"
  mkdir -p "$(dirname "${VENV_ROOT}")"
  python -m venv --clear "${VENV_ROOT}"
fi

source "${VENV_ROOT}/bin/activate"
python -m pip install --upgrade pip wheel

echo "[setup_env] installing project from ${CODE_ROOT}"
pip install --no-cache-dir -e "${CODE_ROOT}[dev]"

# Pre-fetch HuggingFace tokenizers / datasets used by the benchmark to
# keep compute-node start-up fast. Set HF_HOME on scratch so the cache
# is shared between jobs.
export HF_HOME="${SCRATCH}/defaultpp/hf_cache"
mkdir -p "${HF_HOME}"
python - <<'PY'
import os
from transformers import AutoTokenizer
for name in [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "distilbert/distilroberta-base",
    "gpt2",
    "distilgpt2",
    "EleutherAI/gpt-neo-125M",
]:
    print(f"[setup_env] pre-fetch tokenizer: {name}")
    AutoTokenizer.from_pretrained(name, cache_dir=os.environ["HF_HOME"])
PY

echo "[setup_env] done."
