# Shared environment setup for Compute Canada SLURM jobs.
#
# Source this file from every job script:
#
#     source "$SLURM_SUBMIT_DIR/scripts/cc/env.sh"
#
# It loads the StdEnv / Python / CUDA modules, points HuggingFace and
# torch caches at $SCRATCH (so they survive job restarts but do not
# count against $HOME quotas), and activates the project virtual
# environment. The venv is created on first use by setup_env.sh.

set -euo pipefail

# ── Module stack ─────────────────────────────────────────────────────────
module --quiet purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load arrow/15.0.1   # parquet support for dataset_writer

# ── Project paths ────────────────────────────────────────────────────────
: "${PROJECT:?PROJECT is not set; running on a non-CC machine?}"
: "${SCRATCH:?SCRATCH is not set; running on a non-CC machine?}"

PROJECT_ROOT="${PROJECT}/DEFaultplusplus-Transformer-Debugging"
CODE_ROOT="${PROJECT_ROOT}/defaultplusplus"
SCRATCH_ROOT="${SCRATCH}/defaultplusplus"
VENV_ROOT="${SCRATCH}/venvs/defaultplusplus"
RESULTS_ROOT="${PROJECT_ROOT}/results"
BENCH_SHARDS="${SCRATCH_ROOT}/bench_shards"
BENCH_FINAL="${PROJECT_ROOT}/data"

mkdir -p "${SCRATCH_ROOT}" "${BENCH_SHARDS}" "${RESULTS_ROOT}"

# ── HuggingFace / torch caches in scratch ────────────────────────────────
export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${SCRATCH_ROOT}/torch_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# ── Python virtual environment ───────────────────────────────────────────
if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "ERROR: virtual environment ${VENV_ROOT} not found." >&2
  echo "Run scripts/cc/setup_env.sh once on the login node first." >&2
  exit 2
fi
source "${VENV_ROOT}/bin/activate"

# Make the project package importable from the venv.
export PYTHONPATH="${CODE_ROOT}/src:${PYTHONPATH:-}"

echo "[env] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[env] CODE_ROOT=${CODE_ROOT}"
echo "[env] VENV=${VENV_ROOT}"
echo "[env] HF_HOME=${HF_HOME}"
echo "[env] BENCH_SHARDS=${BENCH_SHARDS}"
