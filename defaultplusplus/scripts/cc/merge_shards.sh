#!/usr/bin/env bash
#SBATCH --job-name=defaultpp-merge
#SBATCH --account=def-yourgroup
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Concatenate per-task shards into the final DEFault-bench files.
# Produces:
#   $BENCH_FINAL/encoder_dataset.csv
#   $BENCH_FINAL/decoder_dataset.csv
# and the matching parquet copies.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/cc/env.sh"

cd "${CODE_ROOT}"
python -m defaultplusplus.benchmark.merge \
    --shards-dir "${BENCH_SHARDS}" \
    --out-dir "${BENCH_FINAL}"

echo "[merge] dataset assembled in ${BENCH_FINAL}"
