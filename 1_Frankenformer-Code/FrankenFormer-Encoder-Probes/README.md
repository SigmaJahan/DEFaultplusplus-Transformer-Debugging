# FrankenFormer: Structural Probes for Transformer Fault Detection

Fault injection and structural probe framework implementing the **DEFault++** feature extraction pipeline and **FrankenFormer** fault taxonomy. This repository targets **encoder** transformer models. A companion repository ([FrankenFormer-Decoder-Probes](../FrankenFormer-Decoder-Probes)) provides the equivalent framework for decoder-only models.

## Overview

FrankenFormer systematically injects architectural faults into pretrained transformers, extracts fixed-length feature vectors via structural probes, and detects faults using statistical kill functions. The pipeline produces labeled datasets for training downstream fault classifiers.

### Encoder vs Decoder

| | **Encoder (this repo)** | **Decoder** ([companion repo](../FrankenFormer-Decoder-Probes)) |
|---|---|---|
| **Models** | BERT, DistilBERT, RoBERTa, ELECTRA | GPT-2, DistilGPT2, GPT-Neo-125M |
| **HF API** | `AutoModelForSequenceClassification` | `AutoModelForCausalLM` |
| **Attention** | Bidirectional (full attention) | Causal (triangular mask) |
| **Tasks** | SST-2, MNLI, QQP (GLUE); CoNLL-2003 (NER) | WikiText-2, Lambada (LM); HellaSwag, PIQA, ARC (MC) |
| **Eval Metrics** | Accuracy, F1, AUC-ROC, MLM perplexity | Perplexity, generation quality, cache correctness |
| **Unique Faults** | Pooler faults (CLS pooler scale/zero/noise) | Decoder masking (break causal), KV-cache faults |
| **Kill K_task** | Accuracy drop >= threshold | log-perplexity increase >= log(1.05) |
| **Padding** | Right-padding | Left-padding |
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
| **C_train = 11** | - | Embedding norm/var, positional sensitivity, loss, GNS, step time, confidence, entropy, margin |
| **A_e = 3** | 3 | Epoch aggregation: mean, variance, burst (95th percentile) |
| **C_eval = 2** | - | Accuracy + ECE |
| **A_p = 5** | 5 | Phase aggregation: early_mean, mid_mean, late_mean, slope, final |

## Fault Taxonomy

### Encoder: 12 categories, 62 configurations

| ID | Category | Faults | Severity Params | Detection Metrics |
|----|----------|--------|-----------------|-------------------|
| E1 | Masking | zero_mask, inverted_mask, wrong_broadcast | Binary | mass_pad, cross_example_leak |
| E2 | QKV | zero_q/k/v, swap_qk, tie_heads | Binary | head_similarity, qkv_align |
| E3 | Score | missing_scaling, wrong_scaling | factor: 2x-10x | pre_softmax_score, entropy |
| E4 | Positional | missing, off_by_one, truncate, double | shift: 1-5 | positional_accuracy_delta |
| E5 | Kernel | force_unoptimized, wrong_dropout | dropout: 0.1-0.3 | step_time, memory |
| E6 | Variant | single_head, causal_in_bidirectional | Binary | attention_rank |
| E8 | Embedding | zero, swap, type_drop | fraction: 2-15% | embedding_norm |
| E9 | FFN | weight_scaling, neuron_drop, activation | alpha: 0.3-0.8 | ffn_delta |
| E10 | LayerNorm | gamma_scale, beta_shift, stats | gamma: 0.3-0.8 | ln_std |
| E11 | Residual | drop, scale, noise | alpha: 0.3-0.8 | residual_cos |
| E12 | Output | scale, row_drop, noise | alpha: 0.3-0.8 | logit_entropy, ece |
| E13 | Pooler | scale, zero, noise | alpha: 0.3-0.5 | cls_representation_norm |

### Decoder: 13 categories, 82 configurations

Same as encoder minus Pooler (E13), plus:

| ID | Category | Faults | Detection Metrics |
|----|----------|--------|-------------------|
| E7 | KV Cache | stale, off_by_one, truncate, leak | cache_correctness |
| - | Decoder Masking | break_causal, over_mask, pad_error | mass_future |

## Directory Structure

```
config/
  matrix_encoder.yaml             Full experiment grid (3 models x 2 tasks x 2 seeds)
  local_smoke_matrix.yaml         Minimal local test
  pipeline_configs_probes.json    Stratified fault configurations
  silent_faults_severity.json     62 faults with severity parameters
  smoke_test_pipeline.json        Single baseline config

scripts/
  run_pipeline.py                 Main pipeline runner
  validate_pipeline.py            Validation tests
  smoke_test_probes.py            Probe unit tests
  cache-models-datasets.py        Pre-cache HF models/datasets
  run_local_smoke.sh              Local GPU smoke test
  submit_encoder_probes.sh        SLURM submission (array jobs)
  submit_smoke_test.sh            SLURM smoke test submission
  env_config.sh                   Compute Canada environment vars
  setup.sh                        Cluster venv setup
  pack_venv.sh                    Package venv for SLURM_TMPDIR

src/
  models/
    base_model.py                 BaseModelWrapper + FaultInjectorMixin
    model_wrapper.py              AutoModelForSequenceClassification wrapper
  metrics/
    base_metrics.py               Per-layer structural probes, attention metrics
    metric_collector.py           Batch/epoch aggregation, CKA drift
    classification_metrics.py     Encoder metrics 20-26 (accuracy, F1, AUC, MLM)
    statistics.py                 Welford stats, epoch/window aggregation
    running_metrics.py            Rolling-window statistics
  faults/                         12 fault categories (39 root causes)
    pooler_faults.py              Encoder-specific CLS pooler faults
  kill_functions/                 Kill criteria, permutation tests
  pipeline/
    trainer.py                    Training loop (classification + NER)
    kill_evaluator_csv.py         CSV export of kill evaluations
  utils/
    data_loader.py                GLUE (SST-2, MNLI, QQP, etc.) + CoNLL-2003
    storage.py                    HDF5 metrics + SQLite metadata
    config_manager.py             YAML config loading
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

# Run smoke test (distilbert, sst2, 1 epoch)
python scripts/run_pipeline.py \
  --matrix-config config/local_smoke_matrix.yaml \
  --fault-config config/smoke_test_pipeline.json \
  --results-dir results/local_smoke \
  --cuda

# Check results
ls results/local_smoke/
```

## HPC (Compute Canada)

```bash
bash scripts/setup.sh
vi scripts/env_config.sh              # Set account/paths
source scripts/env_config.sh
python scripts/cache-models-datasets.py --matrix-config config/matrix_encoder.yaml
bash scripts/pack_venv.sh
sbatch scripts/submit_smoke_test.sh    # 1 GPU, 30 min
sbatch scripts/submit_encoder_probes.sh # 18 array jobs, 24h each
```

## Kill Functions

Two complementary criteria:

- **K_beh**: Exact sign-flip permutation test on paired (clean, faulty) metric deltas across seeds. p <= 0.05.
- **K_task (Encoder)**: Accuracy drop >= configurable threshold.
- **K_task (Decoder)**: Mean log-perplexity increase >= log(1.05).

All fault categories have dedicated kill criteria classes routing through `create_kill_criteria()`.
