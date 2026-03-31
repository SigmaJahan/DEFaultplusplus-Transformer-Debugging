# Hierarchical Fault Diagnosis with FPG-Based Explainability for Transformers

NeurIPS 2026 submission — replication package.

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
bash scripts/setup.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Add data files

Place the 4 required CSV files in `data/`. See [`data/README.md`](data/README.md) for file descriptions, checksums, and sources.

```bash
make data-check  # verify all files are present
```

## Reproduction

### Quick smoke test (1 epoch)

```bash
python hierarchical_graph_category_rootcause/train.py --arch encoder --epochs 1
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
python hierarchical_graph_category_rootcause/train.py --arch encoder

# Run all ablation variants
python hierarchical_graph_category_rootcause/evaluate.py --arch both

# Run baselines
python baselines/run_baselines.py --arch both
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
    loader.py                  # CSV loading
    feature_processor.py       # 6-step processing pipeline
    feature_groups.py          # Maps features to FPG component groups (13 groups)
    fundamental_fpg.py         # Fault Propagation Graph (8 deterministic rules)
  models/
    group_encoder.py           # GroupEncoder, GraphAggregator, FlatEncoder

hierarchical_graph_category_rootcause/
    model.py                   # HierarchicalDiagnosisModel + prototypical explainability
    losses.py                  # Detection, category, root-cause, intra-family contrastive
    train.py                   # Training with hierarchical logic
    evaluate.py                # Ablation runner + plot generation
    posthoc_analysis.py        # Group importance analysis
    plotting.py                # NeurIPS-quality figures

baselines/
    run_baselines.py           # DEFault-style, DeepFD-style, AutoTrainer, DeepDiagnosis

configs/base.yaml              # Hyperparameters and data paths
data/                          # CSV data files (not tracked — see data/README.md)
results/                       # Experiment outputs (regenerated)
examples/                      # Motivating example
docs/                          # Thesis chapter and figures
```

## Ablation Variants

| Variant | Graph | Contrastive | Tests |
|---------|-------|-------------|-------|
| basic | — | — | Hierarchical framework alone |
| graph | FPG | — | + graph-conditioned encoding |
| sibling | — | intra-family | + contrastive loss for root-cause separation |
| full | FPG | intra-family | Complete method |

## Motivating Example

See [`examples/motivating_example.py`](examples/motivating_example.py) — a real QKV fusion bug that silently degrades model quality.

## Citation

```
@inproceedings{jahan2026hierarchical,
  title={Hierarchical Fault Diagnosis with FPG-Based Explainability for Transformers},
  author={Jahan, Sigma and others},
  booktitle={NeurIPS},
  year={2026}
}
```
