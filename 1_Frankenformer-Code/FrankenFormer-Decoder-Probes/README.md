# FrankenFormer: Structural Probes for Transformer Fault Detection

Fault injection and structural probe framework implementing the **DEFault++** feature extraction pipeline and **FrankenFormer** fault taxonomy. This repository targets **decoder-only** transformer models. A companion repository ([FrankenFormer-Encoder-Probes](../FrankenFormer-Encoder-Probes)) provides the equivalent framework for encoder models.

## Overview

FrankenFormer systematically injects architectural faults into pretrained transformers, extracts fixed-length feature vectors via structural probes, and detects faults using statistical kill functions. The pipeline produces labeled datasets for training downstream fault classifiers.

### Decoder vs Encoder

| | **Decoder (this repo)** | **Encoder** ([companion repo](../FrankenFormer-Encoder-Probes)) |
|---|---|---|
| **Models** | GPT-2, DistilGPT2, GPT-Neo-125M | BERT, DistilBERT, RoBERTa, ELECTRA |
| **HF API** | `AutoModelForCausalLM` | `AutoModelForSequenceClassification` |
| **Attention** | Causal (triangular mask) | Bidirectional (full attention) |
| **Tasks** | WikiText-2, Lambada (LM); HellaSwag, PIQA, ARC (MC) | SST-2, MNLI, QQP (GLUE); CoNLL-2003 (NER) |
| **Eval Metrics** | Perplexity, generation quality, cache correctness | Accuracy, F1, AUC-ROC, MLM perplexity |
| **Unique Faults** | Decoder masking (break causal), KV-cache faults | Pooler faults (CLS pooler scale/zero/noise) |
| **Kill K_task** | log-perplexity increase >= log(1.05) | Accuracy drop >= threshold |
| **Padding** | Left-padding | Right-padding |
| **Feature Vector** | 1390 dimensions | 1390 dimensions |

### Shared Components

Both repos share the same structural probe architecture, statistical tests, storage backend, and feature construction formula. The 11 core fault categories (masking, QKV, score, positional, kernel, variant, embedding, FFN, LayerNorm, residual, output) are identical in both.

## Feature Construction (DEFault++)

Fixed-length feature vector: **1390 dimensions**.

```
d_final = A_p * (A_e * (A_l * C_int + C_opt + C_train) + C_eval)
        = 5  * (3  * (5  * 12   + 21   + 11    ) + 2    )
        = 5  * 278 = 1390
```

| Stage | Factor | Description |
|-------|--------|-------------|
| **C_int = 12** | - | Attention entropy, pad mass, head similarity, pre-softmax score stats, FFN norm, LN stats, residual cosine, inter-layer CKA, QKV alignment (Q-K, Q-V, K-V) |
| **A_l = 5** | 5 | Layer grouping: early_mean, early_std, mid_mean, mid_std, final_layer |
| **C_opt = 21** | - | Per-component gradient norms + update ratios (5 components x 3 + 6 global) |
| **C_train = 11** | - | Embedding norm/var, positional sensitivity, loss, GNS, step time, confidence, entropy, margin, cache similarity, cache divergence |
| **A_e = 3** | 3 | Epoch aggregation: mean, variance, burst (95th percentile) |
| **C_eval = 2** | - | Perplexity + ECE |
| **A_p = 5** | 5 | Phase aggregation: early_mean, mid_mean, late_mean, slope, final |

## Fault Taxonomy

### Decoder: 13 categories, 82 configurations

| ID | Category | Faults | Severity Params | Detection Metrics |
|----|----------|--------|-----------------|-------------------|
| E1 | Masking | zero_mask, inverted_mask, wrong_broadcast | Binary | mass_pad, cross_example_leak |
| E2 | QKV | zero_q/k/v, swap_qk, tie_heads | Binary | head_similarity, qkv_align |
| E3 | Score | missing_scaling, wrong_scaling | factor: 2x-10x | pre_softmax_score, entropy |
| E4 | Positional | missing, off_by_one, truncate | shift: 1-5 | positional_accuracy_delta |
| E5 | Kernel | force_unoptimized, wrong_dropout | dropout: 0.3 | step_time, memory |
| E6 | Variant | single_head, causal_in_noncausal | Binary | attention_rank |
| E7 | KV Cache | stale, off_by_one, truncate, leak | leak: 0.3-0.8 | cache_correctness |
| E8 | Embedding | zero, swap, type_drop | fraction: 2-15% | embedding_norm |
| E9 | FFN | weight_scaling, neuron_drop, activation | alpha: 0.5-0.8 | ffn_delta |
| E10 | LayerNorm | gamma_scale, beta_shift, stats | gamma: 0.3-0.8 | ln_std |
| E11 | Residual | drop, scale, noise | alpha: 0.3-0.8 | residual_cos |
| E12 | Output | scale, row_drop, noise | alpha: 0.5-0.8 | logit_entropy, ece |
| - | Decoder Masking | break_causal, over_mask, pad_error | visibility: 0.1-0.5 | mass_future |

### Encoder: 12 categories, 62 configurations

Same as decoder minus KV Cache (E7) and Decoder Masking, plus:

| ID | Category | Faults | Detection Metrics |
|----|----------|--------|-------------------|
| E13 | Pooler | scale, zero, noise | cls_representation_norm |

## Directory Structure

```
config/
  matrix_336.yaml                 Full experiment grid (3 models x 2 tasks x 2 seeds)
  local_smoke_matrix.yaml         Minimal local test
  pipeline_configs_probes.json    336 stratified fault configurations
  silent_faults_severity.json     82 faults with severity parameters
  smoke_test_pipeline.json        Single baseline config

scripts/
  run_pipeline.py                 Main pipeline runner
  validate_pipeline.py            Validation tests
  smoke_test_probes.py            Probe unit tests
  cache-models-datasets.py        Pre-cache HF models/datasets
  run_local_smoke.sh              Local GPU smoke test
  submit_probes_336.sh            SLURM submission (336 configs)
  submit_smoke_test.sh            SLURM smoke test submission
  env_config.sh                   Compute Canada environment vars
  setup.sh                        Cluster venv setup
  pack_venv.sh                    Package venv for SLURM_TMPDIR

src/
  models/
    base_model.py                 BaseModelWrapper + FaultInjectorMixin
    model_wrapper.py              AutoModelForCausalLM wrapper
  metrics/
    base_metrics.py               Per-layer structural probes, attention metrics
    metric_collector.py           Batch/epoch aggregation, CKA drift
    generation_metrics.py         Decoder metrics 20-26 (repetition, cache)
    statistics.py                 Welford stats, epoch/window aggregation
    running_metrics.py            Rolling-window statistics
  faults/                         13 fault categories (45 root causes)
  kill_functions/                 Kill criteria, permutation tests
  pipeline/
    trainer.py                    Training loop (LM + MC tasks)
    kill_evaluator_csv.py         CSV export of kill evaluations
  utils/
    data_loader.py                WikiText-2, Lambada, HellaSwag, PIQA, ARC
    storage.py                    HDF5 metrics + SQLite metadata
    config_manager.py             YAML config loading
    config_generator.py           Fault configuration matrix generation
    logger.py                     Logging
    profiler.py                   GPU profiling
    reproducibility.py            Seed management
    math_utils.py                 Safe numerical operations
  export/
    json_dataset_builder.py       JSON dataset from HDF5/SQLite
```

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Cache model and dataset
python scripts/cache-models-datasets.py --matrix-config config/local_smoke_matrix.yaml

# Run validation tests
python scripts/validate_pipeline.py --cuda

# Run smoke test (distilgpt2, wikitext-2, 1 epoch)
bash scripts/run_local_smoke.sh

# Check results
ls results/local_smoke/distilgpt2/wikitext-2/
```

## HPC (Compute Canada)

```bash
bash scripts/setup.sh
vi scripts/env_config.sh              # Set account/paths
source scripts/env_config.sh
python scripts/cache-models-datasets.py --matrix-config config/matrix_336.yaml
bash scripts/pack_venv.sh
sbatch scripts/submit_smoke_test.sh    # 1 GPU, 30 min
sbatch scripts/submit_probes_336.sh    # 18 array jobs, 24h each
```

## Kill Functions

Two complementary criteria:

- **K_beh**: Exact sign-flip permutation test on paired (clean, faulty) metric deltas across seeds. p <= 0.05.
- **K_task (Decoder)**: Mean log-perplexity increase >= log(1.05).
- **K_task (Encoder)**: Accuracy drop >= configurable threshold.

All fault categories have dedicated kill criteria classes routing through `create_kill_criteria()`.
