# DEFault++ Runtime v1 Contract

This document is the normative runtime-v1 contract for DEFault++. It separates the current research artifact from the runtime/product boundary and freezes the documentation precedence that future implementation work must follow.

## 1. Scope Boundary

DEFault++ currently spans two different but compatible artifacts:

- The research/reproduction artifact: the Chapter 7/8 benchmark, grouped diagnosis experiments, and CSV-based offline evaluation pipeline.
- The runtime/product artifact: the `src/defaultplusplus/` package that must observe one live run and produce diagnosis-ready telemetry.

Runtime v1 is not the same thing as the Chapter 7/8 paired-delta benchmark pipeline. The paper and the current reproduction package remain the authority for the science, taxonomy, grouped diagnosis rationale, and benchmark construction. Runtime v1 is the engineering contract for a single-run deployment system built from that science.

## 2. Deployment Contract

The deployment boundary is frozen as follows:

- Research and benchmark generation use paired clean/faulty runs, direction-aligned deltas, and aggregate constructions derived from matched executions.
- Runtime v1 uses single-run anomaly encoding against clean reference statistics.

Runtime components must not assume the presence of a matched clean run at inference time. Any runtime feature, preprocessing step, or diagnosis path that depends on direct clean-vs-faulty pairing is out of contract for runtime v1.

## 3. Supported Model Families for v1

Runtime v1 support is defined at the structural family level, not as a promise to support all Hugging Face models.

Supported families:

- BERT-style encoders: BERT, RoBERTa, DistilBERT, and ALBERT-style encoder structures.
- GPT-style decoders: GPT-2, DistilGPT2, and GPT-Neo-style decoder structures.

Unsupported families are out of contract for v1, including encoder-decoder architectures and families that require materially different extraction logic. Unsupported families must fail closed with an explicit unsupported-model result; they must not be silently approximated or treated as supported by default.

## 4. Canonical Runtime-v1 Feature Contract

The runtime-v1 contract freezes the core collector metric keys and key patterns below. Automatically derived aggregation outputs such as `*_mean`, `*_var`, `*_count`, `*_final`, `*_slope`, `val_*`, and windowed summaries are not yet independent source-of-truth schema items unless listed explicitly below.

### Training Dynamics

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `train_loss` | Primary training loss for the current step | both | `exact` | `runtime_v1` |
| `train_learning_rate` | Optimizer LR from the first parameter group | both | `exact` | `runtime_v1` |
| `runtime_step_time` | Wall-clock step duration | both | `exact` | `runtime_v1` |
| `runtime_steps_per_sec` | Inverse throughput proxy from step time | both | `exact` | `runtime_v1` |
| `runtime_memory_alloc_mb` | Allocated accelerator memory | both | `exact` | `runtime_v1` |
| `runtime_memory_reserved_mb` | Reserved accelerator memory | both | `exact` | `runtime_v1` |
| `loss` | Duplicate alias of `train_loss` | both | `exact` | `research_only` |

### Gradient and Update Behavior

Parameter-group patterns are part of the contract:

- `{group}` may be `embedding`, `classifier`, `layer{i}_attention`, `layer{i}_qkv`, `layer{i}_ffn`, or `layer{i}_layernorm`.

| Metric or Pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `grad_norm_total` | Total gradient norm | both | `exact` | `runtime_v1` |
| `grad_abs_min` | Minimum absolute gradient value | both | `exact` | `runtime_v1` |
| `grad_abs_max` | Maximum absolute gradient value | both | `exact` | `runtime_v1` |
| `grad_zero_ratio` | Fraction of near-zero gradient elements | both | `exact` | `runtime_v1` |
| `gradient_vanish` | Vanishing-gradient indicator | both | `exact` | `runtime_v1` |
| `gradient_explode` | Exploding-gradient indicator | both | `exact` | `runtime_v1` |
| `gradient_variance` | Running variance of total gradient norm | both | `exact` | `runtime_v1` |
| `gradient_noise_scale` | Running noise-scale proxy for total gradient norm | both | `exact` | `runtime_v1` |
| `grad_norm_{group}` | Gradient norm per discovered parameter group | both | `exact` | `runtime_v1` |
| `grad_norm_{group}_window_var` | Running variance of `grad_norm_{group}` | both | `exact` | `runtime_v1` |
| `grad_norm_{group}_gns` | Running noise-scale proxy of `grad_norm_{group}` | both | `exact` | `runtime_v1` |
| `update_active_{group}` | Binary activity flag for a group | both | `exact` | `runtime_v1` |
| `update_ratio_{group}` | Relative parameter update magnitude for a group | both | `exact` | `runtime_v1` |
| `update_ratio_total` | Relative parameter update magnitude overall | both | `exact` | `runtime_v1` |

### Attention and Score Signals

Sampled attention-layer patterns are part of the contract:

- `{layer_idx}` is sampled using the runtime inspector strategy.

| Metric or Pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `L{layer_idx}_attention_entropy_mean` | Mean attention entropy in a sampled layer | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_entropy_std` | Entropy spread across heads | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_max_mean` | Mean max attention weight | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_max_std` | Spread of max attention weights | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_sparsity` | Fraction of low-mass attention values | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_weight_magnitude` | Mean attention mass magnitude | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_pad_mean` / `L{layer_idx}_attention_mass_pad_max` | Mass assigned to padded positions | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_leak` / `L{layer_idx}_attention_mass_leak_max` | Cross-example leak mass | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_cross_example_leak` | Cross-example leak indicator | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_future` | Future-position attention mass | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_special_mean` / `L{layer_idx}_attention_mass_special_std` | Special-token attention mass | both | `exact` | `runtime_v1` |
| `L{layer_idx}_head_similarity_mean` / `L{layer_idx}_head_similarity_std` / `L{layer_idx}_head_similarity_max` | Head-pattern similarity | both | `exact` | `runtime_v1` |
| `L{layer_idx}_positional_recv_mean` / `..._var` / `..._skew` / `..._early` / `..._mid` / `..._late` / `..._mid_over_early` / `..._late_over_early` | Positional receiving profile from attention maps | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_score_var` / `L{layer_idx}_attention_score_skew` | Log-prob proxy for score-shape behavior | both | `approximate` | `runtime_v1` |
| `L{layer_idx}_pre_softmax_score_mean` / `..._var` / `..._skew` / `..._kurt` | Reconstructed QK score statistics | both | `reconstructed` | `runtime_v1` |
| `attention_entropy`, `attention_entropy_mean`, `mass_pad`, `mass_leak`, `cross_example_attention`, `attention_mass_future`, `pre_softmax_score_mean`, `pre_softmax_score_var`, `pre_softmax_score_skew`, `pre_softmax_score_kurt`, `head_similarity_mean`, `head_similarity_max` | Global aliases aggregated across sampled layers | both | `exact` except `pre_softmax_*` = `reconstructed` | `runtime_v1` |

### QKV Alignment

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `qkv_alignment_*` | Dedicated QKV alignment metrics | both | `not_available` | `post_v1` |

Runtime v1 does not freeze any standalone `qkv_alignment_*` keys yet. The current runtime surface only includes QKV-related parameter grouping (`layer{i}_qkv`) and reconstructed `pre_softmax_score_*` statistics.

### Structural, FFN, Residual, and LayerNorm Behavior

| Metric or Pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `ffn_delta_l{layer_idx}_mean` | Hidden-state delta across adjacent layers | both | `reconstructed` | `runtime_v1` |
| `residual_cos_l{layer_idx}_mean` | Cosine similarity between adjacent layer states | both | `reconstructed` | `runtime_v1` |
| `ffn_var_ratio_l{layer_idx}` | Output/input variance ratio proxy | both | `reconstructed` | `runtime_v1` |
| `ln_std_l{layer_idx}_mean` | Layer output std proxy | both | `reconstructed` | `runtime_v1` |
| `ln_mean_abs_l{layer_idx}_mean` | Layer output mean-absolute proxy | both | `reconstructed` | `runtime_v1` |
| `ffn_active_dim_frac_l{layer_idx}` | Fraction of active output dimensions | both | `reconstructed` | `runtime_v1` |
| `ffn_out_skew_l{layer_idx}` | Output skewness proxy | both | `reconstructed` | `runtime_v1` |
| `ffn_delta_mean`, `residual_cos_mean`, `ffn_var_ratio_mean`, `ln_std_mean`, `ln_mean_abs_mean`, `ffn_active_dim_frac_mean`, `ffn_out_skew_mean` | Global aggregates over layerwise structural proxies | both | `reconstructed` | `runtime_v1` |
| `embedding_norm_mean`, `embedding_norm_std` | Token embedding norm statistics | both | `exact` | `runtime_v1` |
| `h1_delta_norm_mean` | First-layer hidden drift | both | `reconstructed` | `runtime_v1` |

### Logit and Task Signals

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `accuracy`, `f1_score`, `precision`, `recall` | Task performance metrics | both | `exact` | `runtime_v1` |
| `logit_nan_ratio`, `logit_inf_ratio` | Numerical health of logits | both | `exact` | `runtime_v1` |
| `nll` | Negative log-likelihood / CE loss over logits | both | `exact` | `runtime_v1` |
| `ece` | Expected calibration error | both | `exact` | `runtime_v1` |
| `logit_entropy` | Predictive entropy | both | `exact` | `runtime_v1` |
| `logit_confidence_mean` | Mean max-class confidence | both | `exact` | `runtime_v1` |
| `logit_kl_uniform` | Divergence from a uniform predictive distribution | both | `exact` | `runtime_v1` |
| `logit_margin_mean`, `logit_margin_var`, `logit_margin_p25`, `logit_margin_p50`, `logit_margin_p75`, `logit_margin_min` | Margin-based confidence diagnostics | both | `exact` | `runtime_v1` |

### Positional Behavior

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `positional_accuracy_early`, `positional_accuracy_late`, `positional_accuracy_delta` | Early/late positional accuracy sensitivity | both | `exact` | `runtime_v1` |
| `positional_margin_early`, `positional_margin_late`, `positional_margin_delta` | Early/late positional margin sensitivity | both | `exact` | `runtime_v1` |
| `positional_loss_early`, `positional_loss_late` | Early/late positional loss sensitivity | both | `exact` | `runtime_v1` |

### Cache Behavior

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `cache_hidden_sim` | Consecutive KV-key similarity proxy | decoder | `approximate` | `runtime_v1` |
| `cache_nll_divergence` | Cache-specific NLL divergence | decoder | `not_available` | `post_v1` |

## 5. Source-of-Truth Precedence

Document precedence is frozen as:

1. `2_Frakenformer-DEFaultpp-Manuscript/Chapter_7_8.pdf` for scientific truth.
2. `docs/runtime_v1_contract.md` for the runtime-v1 contract and doc governance.
3. `7_DEFaultpp-code/defaultplusplus_runtime_roadmap.md` for runtime/product architecture and sequencing.
4. `7_DEFaultpp-code/Plan.md` for subordinate backlog and tests.
5. `7_DEFaultpp-code/README.md` for the current research/reproduction artifact.
6. `7_DEFaultpp-code/features.md` for the derivative runtime feature reference.

When documents conflict, the higher-precedence document wins. `features.md` is derivative of this runtime-v1 contract and must not be treated as an independent authority.
