# Compute Canada execution scripts

These scripts run the DEFault++ benchmark construction and the
diagnostic-model training on a Compute Canada cluster (Cedar, Graham,
Narval). They assume:

- the project lives at `$PROJECT/DEFaultplusplus-Transformer-Debugging/`
  on the cluster (`$PROJECT` is an environment variable on every CC
  node and points to the user's project space),
- a Python virtual environment will be created in `$SCRATCH/venvs/defaultpp`
  on first use (the Python venv is built fresh on each compute node
  because compute nodes do not have access to `$HOME`),
- raw HuggingFace caches and intermediate per-config artifacts live
  under `$SCRATCH/defaultpp/`,
- the consolidated benchmark CSV and trained models are copied back
  to `$PROJECT/.../results/` at the end of each pipeline stage.

## Layout

```
scripts/cc/
  env.sh                # module loads + venv activation
  setup_env.sh          # one-shot venv creation
  bench_array.sh        # SLURM array job that builds DEFault-bench
                        # (one configuration per array task)
  merge_shards.sh       # concatenates per-task shards into the
                        # final CSV / Parquet
  train.sh              # trains the diagnostic model after the
                        # benchmark is built
  ablation.sh           # runs the four ablation variants
```

## Pipeline

```bash
# One-time setup (login node).
bash scripts/cc/setup_env.sh

# Stage 1: build DEFault-bench (heavy, GPU array job).
sbatch scripts/cc/bench_array.sh

# After the array completes, merge per-task shards into the final
# dataset.
sbatch scripts/cc/merge_shards.sh

# Stage 2: train the diagnostic model on the assembled dataset.
sbatch scripts/cc/train.sh

# Stage 3 (optional): run the four ablation variants.
sbatch scripts/cc/ablation.sh
```

All scripts are idempotent: they re-resolve absolute paths from
`$SLURM_SUBMIT_DIR` and `$PROJECT` and skip configurations whose status
JSON already records a completed run.
