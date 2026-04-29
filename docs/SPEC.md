# DEFault++ Specification

This is the single specification document for DEFault++. It consolidates
the runtime feature schema, the architectural principles, and the
unfinished roadmap items. Everything else (PyPI install, public API
walk-throughs, research-side commands) lives in the READMEs and
[`defaultplusplus/RESEARCH.md`](../defaultplusplus/RESEARCH.md).

The scientific source of truth is [`../DEFault++.pdf`](../DEFault++.pdf).
This file does not restate it. It pins the engineering contract built
on top of it.

---

## 1. Architectural principles

DEFault++ is a single-run, transformer-specific telemetry and diagnosis
system. Four decisions shape it.

### 1.1 Canonical semantic feature schema

The invariant part of DEFault++ is the *meaning* of the measurements:
attention entropy, padding / future leakage, QKV alignment, pre-softmax
score statistics, gradient and update behavior, FFN output shape,
LayerNorm statistics, residual-stream integrity, output uncertainty,
KV-cache consistency, runtime / memory, and validation quality. The
extraction path may differ across model families; the feature
*definitions* must not. The frozen schema is in Section 2 below.

### 1.2 Family-level adapters, not per-model hooks

Compatibility is defined over structural model families, not by HF
checkpoint name. The two supported families are:

- **BERT-style encoders** — bidirectional attention. BERT, RoBERTa,
  DistilBERT, ALBERT-style encoders.
- **GPT-style decoders** — causal attention. GPT-2, DistilGPT-2,
  GPT-Neo-style decoders.

Other families (encoder-decoder T5 / BART, RetNet, Mamba, MoE) are
out of contract for v1. They must fail closed at construction time
with an explicit unsupported-model error. This is implemented in
`FeatureExtractor.__init__`.

### 1.3 Single-run anomaly encoding, not paired clean-vs-faulty deltas at deployment

Benchmark construction uses paired clean and faulty fine-tuning runs
to generate labeled training examples. That is correct for offline
benchmark generation. It is **not** the runtime contract. The runtime
extractor must turn one user run into anomaly features against a
learned clean reference. Any feature, preprocessing step, or diagnosis
path that depends on a matched clean run at inference time is out of
contract for runtime v1.

### 1.4 HF outputs + targeted hooks + sampled profiler + streaming aggregation

The extraction substrate is fixed:

- HuggingFace `ModelOutput` objects already expose `hidden_states` and
  `attentions` when requested. Use them.
- `Trainer` callbacks are read-only and cannot alter the forward pass.
  Per-step metrics that need attention weights or hidden states must
  be captured through a `compute_loss` override that hands references
  to the callback (see `DEFaultPlusCallback.capture_inputs` and
  `capture_outputs`).
- PyTorch profiler is used only for sampled windows (kernel-time and
  memory features), not as a per-step hook.
- Aggregation runs on a Welford-stable streaming basis so the extractor
  works on long runs without unbounded memory.

---

## 2. Output schema

The output of `FeatureExtractor.finalize()` is a flat
`dict[str, float]` keyed by the names below. Every key is in scope
`runtime_v1` unless explicitly tagged `research_only` or `post_v1`.

### Status legend

- `exact` — measured directly from the runtime signal named by the
  feature.
- `reconstructed` — derived from accessible runtime internals, not
  directly observable in the form named. Example: FFN sublayer
  behavior inferred from adjacent hidden states because the actual
  sublayer boundary is not exposed.
- `approximate` — proxy that captures related behavior but is not a
  faithful one-to-one reconstruction.
- `not_available` — reserved name; explicitly excluded from the
  current runtime surface.

### Naming conventions

- `{group}` is one of `embedding`, `classifier`, `layer{i}_attention`,
  `layer{i}_qkv`, `layer{i}_ffn`, `layer{i}_layernorm`.
- `{layer_idx}` is sampled by the runtime inspector strategy, not
  every layer unconditionally.
- Aggregation outputs (`*_mean`, `*_var`, `*_count`, `*_final`,
  `*_slope`, `val_*`, windowed summaries) are derived from the schema
  below; they do not constitute independent contract entries unless
  listed explicitly.

### 2.1 Training dynamics

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `train_loss` | Primary training loss for the current step | both | `exact` | `runtime_v1` |
| `train_learning_rate` | Optimizer LR from the first parameter group | both | `exact` | `runtime_v1` |
| `runtime_step_time` | Wall-clock step duration | both | `exact` | `runtime_v1` |
| `runtime_steps_per_sec` | Inverse throughput proxy from step time | both | `exact` | `runtime_v1` |
| `runtime_memory_alloc_mb` | Allocated accelerator memory | both | `exact` | `runtime_v1` |
| `runtime_memory_reserved_mb` | Reserved accelerator memory | both | `exact` | `runtime_v1` |
| `loss` | Duplicate alias of `train_loss` | both | `exact` | `research_only` |

### 2.2 Gradient and update behavior

| Metric or pattern | Meaning | Availability | Status | Scope |
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

### 2.3 Attention and score signals

Sampled attention-layer patterns are part of the contract. `{layer_idx}`
is sampled using the runtime inspector strategy.

| Metric or pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `L{layer_idx}_attention_entropy_mean` | Mean attention entropy in a sampled layer | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_entropy_std` | Entropy spread across heads | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_max_mean` | Mean max attention weight | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_max_std` | Spread of max attention weights | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_sparsity` | Fraction of low-mass attention values | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_weight_magnitude` | Mean attention mass magnitude | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_pad_mean` / `..._mass_pad_max` | Mass assigned to padded positions | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_leak` / `..._mass_leak_max` | Cross-example leak mass | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_cross_example_leak` | Cross-example leak indicator | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_future` | Future-position attention mass | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_mass_special_mean` / `..._mass_special_std` | Special-token attention mass | both | `exact` | `runtime_v1` |
| `L{layer_idx}_head_similarity_mean` / `..._std` / `..._max` | Head-pattern similarity | both | `exact` | `runtime_v1` |
| `L{layer_idx}_positional_recv_mean` / `..._var` / `..._skew` / `..._early` / `..._mid` / `..._late` / `..._mid_over_early` / `..._late_over_early` | Positional receiving profile from attention maps | both | `exact` | `runtime_v1` |
| `L{layer_idx}_attention_score_var` / `..._score_skew` | Log-prob proxy for score-shape behavior | both | `approximate` | `runtime_v1` |
| `L{layer_idx}_pre_softmax_score_mean` / `..._var` / `..._skew` / `..._kurt` | Reconstructed QK score statistics | both | `reconstructed` | `runtime_v1` |
| Global aliases: `attention_entropy`, `attention_entropy_mean`, `mass_pad`, `mass_leak`, `cross_example_attention`, `attention_mass_future`, `pre_softmax_score_mean`, `pre_softmax_score_var`, `pre_softmax_score_skew`, `pre_softmax_score_kurt`, `head_similarity_mean`, `head_similarity_max` | Aggregated across sampled layers | both | `exact` (`pre_softmax_*` = `reconstructed`) | `runtime_v1` |

### 2.4 QKV alignment

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `qkv_alignment_*` | Dedicated QKV alignment metrics | both | `not_available` | `post_v1` |

The runtime surface in v1 includes QKV-related parameter grouping
(`layer{i}_qkv`) and reconstructed `pre_softmax_score_*` only. No
dedicated `qkv_alignment_*` keys are frozen yet.

### 2.5 Structural, FFN, residual, LayerNorm behavior

| Metric or pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `ffn_delta_l{layer_idx}_mean` | Hidden-state delta across adjacent layers | both | `reconstructed` | `runtime_v1` |
| `residual_cos_l{layer_idx}_mean` | Cosine similarity between adjacent layer states | both | `reconstructed` | `runtime_v1` |
| `ffn_var_ratio_l{layer_idx}` | Output / input variance ratio proxy | both | `reconstructed` | `runtime_v1` |
| `ln_std_l{layer_idx}_mean` | Layer output std proxy | both | `reconstructed` | `runtime_v1` |
| `ln_mean_abs_l{layer_idx}_mean` | Layer output mean-absolute proxy | both | `reconstructed` | `runtime_v1` |
| `ffn_active_dim_frac_l{layer_idx}` | Fraction of active output dimensions | both | `reconstructed` | `runtime_v1` |
| `ffn_out_skew_l{layer_idx}` | Output skewness proxy | both | `reconstructed` | `runtime_v1` |
| Global aggregates: `ffn_delta_mean`, `residual_cos_mean`, `ffn_var_ratio_mean`, `ln_std_mean`, `ln_mean_abs_mean`, `ffn_active_dim_frac_mean`, `ffn_out_skew_mean` | Aggregated across layerwise structural proxies | both | `reconstructed` | `runtime_v1` |
| `embedding_norm_mean`, `embedding_norm_std` | Token embedding norm statistics | both | `exact` | `runtime_v1` |
| `h1_delta_norm_mean` | First-layer hidden drift | both | `reconstructed` | `runtime_v1` |

### 2.6 Logit and task signals

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

### 2.7 Positional behavior

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `positional_accuracy_early`, `positional_accuracy_late`, `positional_accuracy_delta` | Early / late positional accuracy sensitivity | both | `exact` | `runtime_v1` |
| `positional_margin_early`, `positional_margin_late`, `positional_margin_delta` | Early / late positional margin sensitivity | both | `exact` | `runtime_v1` |
| `positional_loss_early`, `positional_loss_late` | Early / late positional loss sensitivity | both | `exact` | `runtime_v1` |

### 2.8 Cache behavior (decoder only)

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `cache_hidden_sim` | Consecutive KV-key similarity proxy | decoder | `approximate` | `runtime_v1` |
| `cache_nll_divergence` | Cache-specific NLL divergence | decoder | `not_available` | `post_v1` |

---

## 3. Roadmap

This section is the running list of unfinished items. As each is
completed it should be removed from this section and reflected in the
schema above (with the corresponding metric promoted from
`reconstructed` / `approximate` / `not_available` to `exact`).

### 3.1 Schema gaps to close

- **QKV alignment metrics (`qkv_alignment_*`)** — currently
  `not_available` / `post_v1`. The runtime surface needs dedicated
  Q-K, Q-V, K-V cosine-similarity outputs. Gap: hook the post-projection
  Q, K, V tensors at the targeted layer set rather than reconstructing
  them after the fact.
- **`cache_nll_divergence`** — currently `not_available` / `post_v1`.
  Requires a fresh-vs-cached forward at sampled generation steps.
- **Structural proxies promoted to `exact`** — `ffn_delta_*`,
  `residual_cos_*`, `ffn_var_ratio_*`, `ln_std_*`, `ln_mean_abs_*`,
  `ffn_active_dim_frac_*`, `ffn_out_skew_*` are all currently
  `reconstructed` from adjacent hidden states. Promoting them to
  `exact` requires sublayer-boundary hooks rather than hidden-state
  differencing. Tracking issue: add `pre_ffn` / `post_ffn` / `pre_ln`
  / `post_ln` capture sites in `extraction/inspector.py`.
- **`L{layer_idx}_attention_score_*`** — currently `approximate`. The
  log-prob proxy should be replaced by direct pre-softmax score
  statistics once score capture is wired through the attention hook.

### 3.2 Runtime product items

- **Single-run anomaly encoding pipeline** — the offline pipeline
  uses paired clean / faulty deltas. Build a `RuntimeNormalizer` that
  loads a learned clean reference and converts a single live run's
  feature vector into the same shape the diagnostic model expects.
  Lives in `defaultplusplus/processing/` (currently a reserved
  namespace).
- **Pretrained diagnostic-model weights** — train and ship the
  encoder + decoder diagnostic models as the v1 release blob. Code
  drop into `defaultplusplus/pretrained/weights/` plus a
  `defaultplusplus.diagnosis.load_pretrained()` accessor.
- **`MetricCollector.feature_names` as a canonical contract** — the
  property currently returns whatever was emitted in the last epoch.
  Promote it to a frozen, schema-checked list so a downstream
  classifier can pin the input dimensionality at training time.

### 3.3 Benchmark items

- **`make benchmark`** — wire the new `defaultplusplus.benchmark`
  pipeline through to a single command that produces `data/*.csv` from
  scratch, replacing the legacy FrankenFormer probe code.
- **Operator implementation directory** — the `OPERATORS` catalog
  names 45 mutations but the per-operator injection classes still need
  to be split into `defaultplusplus/deform/operator_impls/<id>.py` so
  the benchmark runner can resolve injectors by ID without a custom
  `injector_factory` callable.

### 3.4 Out of v1 scope

- Encoder-decoder architectures (T5, BART). They require two-stream
  attention metrics that the current schema does not represent.
- Sparse-attention variants (Longformer, BigBird) and Mixture-of-Experts
  routing diagnostics.
- Distributed-training-specific signals (gradient sync drift,
  parameter-server staleness).

---

## 4. Versioning

Schema version follows the package version (`defaultplusplus.__version__`).

- A **MAJOR** bump removes or renames a `runtime_v1` entry. This
  invalidates pre-trained diagnostic-model weights.
- A **MINOR** bump promotes an entry from `reconstructed` /
  `approximate` / `not_available` to `exact`, or adds a new entry
  whose absence would have been silently ignored.
- A **PATCH** bump fixes a metric implementation without changing its
  output contract.

Changes to this document must be logged in
[`../defaultplusplus/CHANGELOG.md`](../defaultplusplus/CHANGELOG.md).
