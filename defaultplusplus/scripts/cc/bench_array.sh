#!/usr/bin/env bash
#SBATCH --job-name=defaultpp-bench
#SBATCH --account=def-yourgroup
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3738%64
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# DEFault-bench construction. One SLURM array task per fault
# configuration. Each task runs paired clean / faulty fine-tuning over
# five matched seeds, applies the sign-flip permutation test, and
# appends the resulting labeled instance to a per-task CSV shard. A
# separate merge_shards job concatenates the shards into the final
# benchmark file.
#
# Edit --account, --array, and --time to match your allocation. The
# upper end of --array should be N_CONFIGURATIONS - 1, and %64 caps
# concurrent tasks to keep within the per-user job limit.

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/cc/env.sh"

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_FILE="${BENCH_SHARDS}/shard_${ARRAY_INDEX:0>5}.csv"
STATUS_DIR="${BENCH_SHARDS}/status_${ARRAY_INDEX:0>5}"
mkdir -p "${STATUS_DIR}"

echo "[bench] array index ${ARRAY_INDEX}"
echo "[bench] shard       ${SHARD_FILE}"
echo "[bench] status      ${STATUS_DIR}"

cd "${CODE_ROOT}"
python -m defaultplusplus.benchmark \
    --array-index "${ARRAY_INDEX}" \
    --shard "${SHARD_FILE}" \
    --status-dir "${STATUS_DIR}"

echo "[bench] done index ${ARRAY_INDEX}"
