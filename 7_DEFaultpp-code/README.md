# Hierarchical Fault Diagnosis with FPG-Based Explainability for Transformers

This directory contains the active DEFault++ package and experiment code. The repository is no longer flat: mutation data, baselines, manuscript assets, and user-study material now live in sibling top-level directories.

This is the canonical research/reproduction README for the current artifact and replication package. The runtime library is under active construction in `src/defaultplusplus/`; runtime design authority lives in [`../docs/runtime_v1_contract.md`](../docs/runtime_v1_contract.md) and [`defaultplusplus_runtime_roadmap.md`](defaultplusplus_runtime_roadmap.md).

## Problem

Transformer models can contain subtle implementation faults — bugs in attention computation, layer normalization, positional encoding, etc. — that produce models which train and run but silently underperform. Diagnosing *which specific bug* caused a failure is hard because (a) there are dozens of possible root causes, (b) many root causes produce similar symptoms, and (c) faults propagate across interconnected components.

## Our Approach

We decompose fault diagnosis into three hierarchical stages:

```
Stage 1                    Stage 2                     Stage 3
Fault Detection     ->     Fault Categorization  ->    Root-Cause Diagnosis
Is something wrong?        Which subsystem?            Which exact bug?
(clean vs faulty)          (e.g., qkv, ffn, ln)       (e.g., zero_query)
```

### Three Contributions

1. **Hierarchical Diagnosis Framework** — Decompose flat multi-class classification (38–40 root causes) into three stages with stage-specific losses, reducing effective classification difficulty at each level.

2. **Intra-Family Contrastive Loss** — Pull embeddings of the same root cause together and push apart embeddings of different root causes within each fault family, targeting the hardest distinctions.

3. **FPG-Based Explainability** — Every diagnosis comes with a built-in explanation via group-structured embeddings aligned to Fault Propagation Graph (FPG) nodes. Distance to each root-cause prototype decomposes by transformer component.

## Setup

### 1. Create virtual environment

```bash
bash ../scripts/setup.sh
source ../.venv/bin/activate
```

Or manually:

```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -e ".[dev]"
```

### 2. Add data files

The tracked mutation datasets live in `../3_Mutation-Data-from-Frakenformer/`. They should not be duplicated into this directory.

```bash
make data-check  # verify all files are present
```

## Reproduction

### Quick smoke test (1 epoch)

```bash
python -m hierarchical_graph_category_rootcause.train --arch encoder --epochs 1
```

### Full experiments

```bash
# Full training (both architectures, full method)
make train

# All 4 ablation variants x 2 architectures x 5 folds
make ablation

# All baselines (DEFault-style, DeepFD-style, AutoTrainer-style, DeepDiagnosis-style)
make baselines

# Everything in sequence
make all
```

### Individual commands

```bash
# Train full method on encoder only
python -m hierarchical_graph_category_rootcause.train --arch encoder

# Run all ablation variants
python -m hierarchical_graph_category_rootcause.evaluate --arch both

# Run baselines
python ../4_Baseline-comparison_with_defaultpp/run_baselines.py --arch both
```

## Data

Mutation testing benchmarks from the FrankenFormer project. Faults are injected into transformer implementations; the mutation testing framework determines whether each fault was **killed** (detected) or **survived**.

| | Encoder | Decoder |
|---|---|---|
| Samples | 9,560 | 9,310 |
| Clean | 735 (7.7%) | 5,215 (56.0%) |
| Faulty (killed) | 8,825 (92.3%) | 4,095 (44.0%) |
| Fault categories | 11 | 12 |
| Root causes | 38 | 40 |

## Evaluation

| Stage | Metric | What it measures |
|-------|--------|------------------|
| 1. Detection | AUROC, F1 | Can we tell clean from faulty? |
| 2. Categorization | Macro-F1 | Can we identify the fault family? |
| 3. Root cause (oracle) | Macro-F1 | Given correct family, can we diagnose the root cause? |
| 3. Root cause (end-to-end) | Macro-F1 | Full pipeline using predicted family |

5-fold GroupKFold (no model/dataset/seed overlap between train and test), 80/20 stratified train/val split within each fold.

## Project Structure

```
src/
  data/
  defaultplusplus/
  models/

hierarchical_graph_category_rootcause/
configs/base.yaml

../3_Mutation-Data-from-Frakenformer/    # shared mutation datasets
../4_Baseline-comparison_with_defaultpp/ # baseline comparison scripts
../2_Frakenformer-DEFaultpp-Manuscript/  # manuscript assets and figures
../results/                              # generated outputs
```

## Ablation Variants

| Variant | Graph | Contrastive | Tests |
|---------|-------|-------------|-------|
| basic | — | — | Hierarchical framework alone |
| graph | FPG | — | + graph-conditioned encoding |
| sibling | — | intra-family | + contrastive loss for root-cause separation |
| full | FPG | intra-family | Complete method |

## Motivating Example

See [`qkv_fusion_bug.py`](../2_Frakenformer-DEFaultpp-Manuscript/defaultpp/qkv_fusion_bug.py) for a motivating QKV fusion bug example.

## Citation

```
@inproceedings{jahan2026hierarchical,
  title={Hierarchical Fault Diagnosis with FPG-Based Explainability for Transformers},
  author={Jahan, Sigma and others},
  booktitle={NeurIPS},
  year={2026}
}
```
