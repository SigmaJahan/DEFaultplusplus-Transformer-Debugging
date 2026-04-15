# DEFault++ Runtime Feature Reference

This document is the runtime feature reference for DEFault++. It is not the scientific source of truth and it is not an independent schema authority. The normative source for runtime-v1 scope, feature status, and document precedence is [`../docs/runtime_v1_contract.md`](../docs/runtime_v1_contract.md). The scientific source of truth remains `../2_Frakenformer-DEFaultpp-Manuscript/Chapter_7_8.pdf`.

This file intentionally documents only the frozen runtime-v1 core and explicitly marks deferred or unavailable material. It does not claim completeness beyond that scope.

## Labels

### Status

- `exact`: measured directly from the runtime signal named by the feature.
- `reconstructed`: derived from accessible runtime internals but not directly exposed in the exact form implied by the Chapter 7/8 benchmark.
- `approximate`: proxy metric that captures related behavior but is not a faithful one-to-one reconstruction of the research signal.
- `not_available`: reserved, known missing, or intentionally excluded from the current runtime surface.

### Scope

- `runtime_v1`: part of the frozen runtime-v1 contract.
- `research_only`: present in current code or historical pipelines, but not part of the runtime-v1 contract.
- `post_v1`: explicitly deferred until after runtime-v1 stabilization.

## Naming Conventions

- `{group}` may be `embedding`, `classifier`, `layer{i}_attention`, `layer{i}_qkv`, `layer{i}_ffn`, or `layer{i}_layernorm`.
- `{layer_idx}` refers to sampled runtime inspection layers, not every layer unconditionally.
- Automatically derived aggregation keys such as `*_mean`, `*_var`, `*_count`, `*_final`, `*_slope`, `val_*`, and windowed summaries are not separately authoritative unless listed in this document.

## Group 1: Training Dynamics

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `train_loss` | Primary training loss for the current step | encoder, decoder | `exact` | `runtime_v1` |
| `train_learning_rate` | Learning rate from the first optimizer parameter group | encoder, decoder | `exact` | `runtime_v1` |
| `runtime_step_time` | Wall-clock step duration | encoder, decoder | `exact` | `runtime_v1` |
| `runtime_steps_per_sec` | Throughput proxy computed from step time | encoder, decoder | `exact` | `runtime_v1` |
| `runtime_memory_alloc_mb` | Allocated accelerator memory | encoder, decoder | `exact` | `runtime_v1` |
| `runtime_memory_reserved_mb` | Reserved accelerator memory | encoder, decoder | `exact` | `runtime_v1` |
| `loss` | Duplicate alias of `train_loss` | encoder, decoder | `exact` | `research_only` |

## Group 2: Gradient and Update Behavior

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `grad_norm_total` | Total gradient norm | encoder, decoder | `exact` | `runtime_v1` |
| `grad_abs_min` | Minimum absolute gradient magnitude | encoder, decoder | `exact` | `runtime_v1` |
| `grad_abs_max` | Maximum absolute gradient magnitude | encoder, decoder | `exact` | `runtime_v1` |
| `grad_zero_ratio` | Fraction of near-zero gradient elements | encoder, decoder | `exact` | `runtime_v1` |
| `gradient_vanish` | Vanishing-gradient indicator | encoder, decoder | `exact` | `runtime_v1` |
| `gradient_explode` | Exploding-gradient indicator | encoder, decoder | `exact` | `runtime_v1` |
| `gradient_variance` | Running variance of total gradient norm | encoder, decoder | `exact` | `runtime_v1` |
| `gradient_noise_scale` | Running noise-scale proxy of total gradient norm | encoder, decoder | `exact` | `runtime_v1` |
| `grad_norm_{group}` | Gradient norm for a discovered parameter group | encoder, decoder | `exact` | `runtime_v1` |
| `grad_norm_{group}_window_var` | Running variance of `grad_norm_{group}` | encoder, decoder | `exact` | `runtime_v1` |
| `grad_norm_{group}_gns` | Running noise-scale proxy of `grad_norm_{group}` | encoder, decoder | `exact` | `runtime_v1` |
| `update_active_{group}` | Binary activity flag for a parameter group | encoder, decoder | `exact` | `runtime_v1` |
| `update_ratio_{group}` | Relative parameter update magnitude for a parameter group | encoder, decoder | `exact` | `runtime_v1` |
| `update_ratio_total` | Relative parameter update magnitude overall | encoder, decoder | `exact` | `runtime_v1` |

## Group 3: Attention and Score Signals

### Per-sampled-layer patterns

| Feature Pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `L{layer_idx}_attention_entropy_mean` | Mean attention entropy | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_entropy_std` | Entropy spread across heads | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_max_mean` | Mean maximum attention weight | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_max_std` | Spread of maximum attention weights | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_sparsity` | Fraction of low-mass attention entries | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_weight_magnitude` | Mean attention magnitude | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_pad_mean`, `L{layer_idx}_attention_mass_pad_max` | Attention mass assigned to padded positions | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_leak`, `L{layer_idx}_attention_mass_leak_max` | Cross-example leak mass | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_cross_example_leak` | Cross-example leak indicator | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_future` | Future-position attention mass | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_special_mean`, `L{layer_idx}_attention_mass_special_std` | Special-token attention mass | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_head_similarity_mean`, `L{layer_idx}_head_similarity_std`, `L{layer_idx}_head_similarity_max` | Similarity among attention heads | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_positional_recv_mean`, `L{layer_idx}_positional_recv_var`, `L{layer_idx}_positional_recv_skew`, `L{layer_idx}_positional_recv_early`, `L{layer_idx}_positional_recv_mid`, `L{layer_idx}_positional_recv_late`, `L{layer_idx}_positional_recv_mid_over_early`, `L{layer_idx}_positional_recv_late_over_early` | Positional receiving profile from attention maps | encoder, decoder | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_score_var`, `L{layer_idx}_attention_score_skew` | Score-shape proxy from log-prob space | encoder, decoder | `approximate` | `runtime_v1` |
| `L{layer_idx}_pre_softmax_score_mean`, `L{layer_idx}_pre_softmax_score_var`, `L{layer_idx}_pre_softmax_score_skew`, `L{layer_idx}_pre_softmax_score_kurt` | Reconstructed pre-softmax QK score statistics | encoder, decoder | `reconstructed` | `runtime_v1` |

### Global aliases across sampled layers

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `attention_entropy`, `attention_entropy_mean` | Aggregate attention entropy | encoder, decoder | `exact` | `runtime_v1` |
| `mass_pad` | Maximum sampled-layer padding mass | encoder, decoder | `exact` | `runtime_v1` |
| `mass_leak` | Maximum sampled-layer leak mass | encoder, decoder | `exact` | `runtime_v1` |
| `cross_example_attention` | Maximum sampled-layer cross-example leak indicator | encoder, decoder | `exact` | `runtime_v1` |
| `attention_mass_future` | Maximum sampled-layer future attention mass | encoder, decoder | `exact` | `runtime_v1` |
| `pre_softmax_score_mean`, `pre_softmax_score_var`, `pre_softmax_score_skew`, `pre_softmax_score_kurt` | Aggregate reconstructed pre-softmax score statistics | encoder, decoder | `reconstructed` | `runtime_v1` |
| `head_similarity_mean`, `head_similarity_max` | Aggregate head similarity | encoder, decoder | `exact` | `runtime_v1` |

## Group 4: QKV Alignment

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `qkv_alignment_*` | Dedicated QKV alignment metrics | encoder, decoder | `not_available` | `post_v1` |

Runtime v1 does not currently freeze any standalone `qkv_alignment_*` keys. The current runtime surface only includes QKV-related parameter grouping (`layer{i}_qkv`) and reconstructed `pre_softmax_score_*` statistics.

## Group 5: Structural, FFN, Residual, and LayerNorm Behavior

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `ffn_delta_l{layer_idx}_mean` | Adjacent-layer hidden-state delta proxy | encoder, decoder | `reconstructed` | `runtime_v1` |
| `residual_cos_l{layer_idx}_mean` | Adjacent-layer residual cosine proxy | encoder, decoder | `reconstructed` | `runtime_v1` |
| `ffn_var_ratio_l{layer_idx}` | Output/input variance ratio proxy | encoder, decoder | `reconstructed` | `runtime_v1` |
| `ln_std_l{layer_idx}_mean` | Layer output std proxy | encoder, decoder | `reconstructed` | `runtime_v1` |
| `ln_mean_abs_l{layer_idx}_mean` | Layer output mean-absolute proxy | encoder, decoder | `reconstructed` | `runtime_v1` |
| `ffn_active_dim_frac_l{layer_idx}` | Fraction of active output dimensions | encoder, decoder | `reconstructed` | `runtime_v1` |
| `ffn_out_skew_l{layer_idx}` | Output skewness proxy | encoder, decoder | `reconstructed` | `runtime_v1` |
| `ffn_delta_mean`, `residual_cos_mean`, `ffn_var_ratio_mean`, `ln_std_mean`, `ln_mean_abs_mean`, `ffn_active_dim_frac_mean`, `ffn_out_skew_mean` | Global structural aggregates | encoder, decoder | `reconstructed` | `runtime_v1` |
| `embedding_norm_mean`, `embedding_norm_std` | Token embedding norm statistics | encoder, decoder | `exact` | `runtime_v1` |
| `h1_delta_norm_mean` | First-layer hidden drift proxy | encoder, decoder | `reconstructed` | `runtime_v1` |

## Group 6: Logit and Task Signals

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `accuracy`, `f1_score`, `precision`, `recall` | Task performance metrics | encoder, decoder | `exact` | `runtime_v1` |
| `logit_nan_ratio`, `logit_inf_ratio` | Numerical health of logits | encoder, decoder | `exact` | `runtime_v1` |
| `nll` | Negative log-likelihood / cross-entropy on logits | encoder, decoder | `exact` | `runtime_v1` |
| `ece` | Expected calibration error | encoder, decoder | `exact` | `runtime_v1` |
| `logit_entropy` | Predictive entropy | encoder, decoder | `exact` | `runtime_v1` |
| `logit_confidence_mean` | Mean maximum-class confidence | encoder, decoder | `exact` | `runtime_v1` |
| `logit_kl_uniform` | Divergence from a uniform predictive distribution | encoder, decoder | `exact` | `runtime_v1` |
| `logit_margin_mean`, `logit_margin_var`, `logit_margin_p25`, `logit_margin_p50`, `logit_margin_p75`, `logit_margin_min` | Margin-based confidence diagnostics | encoder, decoder | `exact` | `runtime_v1` |

## Group 7: Positional Behavior

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `positional_accuracy_early`, `positional_accuracy_late`, `positional_accuracy_delta` | Early/late positional accuracy sensitivity | encoder, decoder | `exact` | `runtime_v1` |
| `positional_margin_early`, `positional_margin_late`, `positional_margin_delta` | Early/late positional margin sensitivity | encoder, decoder | `exact` | `runtime_v1` |
| `positional_loss_early`, `positional_loss_late` | Early/late positional loss sensitivity | encoder, decoder | `exact` | `runtime_v1` |

## Group 8: Cache Behavior

| Feature | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `cache_hidden_sim` | Consecutive KV-key similarity proxy | decoder only | `approximate` | `runtime_v1` |
| `cache_nll_divergence` | Cache-specific NLL divergence | decoder only | `not_available` | `post_v1` |

## Excluded from Runtime v1

The following are intentionally not part of the frozen runtime-v1 core:

- Curvature and Hessian-related metrics such as `sharpness_*` and `hessian_*`: `post_v1`
- Activation-magnitude feature groups and similar expansion work: `post_v1`
- Dedicated `qkv_alignment_*` metrics: `post_v1`
- `cache_nll_divergence`: `not_available`, kept only as an explicit missing slot
- The duplicate `loss` alias: `research_only`, not part of the stable runtime-v1 contract
- Automatically derived validation/window features such as `val_*`, `*_early_mean`, `*_mid_mean`, `*_late_mean`, `*_slope`, and `*_final`: not independently frozen as runtime-v1 schema items
