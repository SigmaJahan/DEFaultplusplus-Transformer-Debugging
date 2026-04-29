#!/usr/bin/env bash
#SBATCH --job-name=defaultpp-train
#SBATCH --account=def-yourgroup
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Train the diagnostic model on the assembled DEFault-bench. Runs the
# full method (FPG message passing + separation loss) for both
# architectures.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/cc/env.sh"

cd "${CODE_ROOT}"
python -m hierarchical_graph_category_rootcause.train \
    --arch both \
    --output "${RESULTS_ROOT}/hierarchical_graph_category_rootcause"

echo "[train] done; results under ${RESULTS_ROOT}"
