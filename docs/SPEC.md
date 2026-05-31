# DEFault++ Specification

This is the single specification document for DEFault++. It consolidates
the runtime feature schema, the architectural principles, and the scope
of v1 with its planned extensions. Everything else (PyPI install, public
API walk-throughs, research-side commands) lives in the READMEs and
[`defaultplusplus/RESEARCH.md`](../defaultplusplus/RESEARCH.md).

The scientific reference is the DEFault++ manuscript (in preparation;
code archived at [doi:10.5281/zenodo.20019817](https://doi.org/10.5281/zenodo.20019817)).
This file does not restate it. It pins the engineering API built on top
of it. For the method itself, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 1. Architectural principles

DEFault++ is a single-run, transformer-specific telemetry and diagnosis
system. Four decisions shape it.

### 1.1 Stable semantic feature schema

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
out of scope for v1. They must fail closed at construction time
with an explicit unsupported-model error. This is implemented in
`FeatureExtractor.__init__`.

### 1.3 Single-run anomaly encoding, not paired clean-vs-faulty deltas at deployment

Benchmark construction uses paired clean and faulty fine-tuning runs
to generate labeled training examples. That is correct for offline
benchmark generation. It is **not** the runtime API. The runtime
extractor must turn one user run into anomaly features against a
learned clean reference. Any feature, preprocessing step, or diagnosis
path that depends on a matched clean run at inference time is out of
scope for runtime v1.

### 1.4 HF outputs + targeted hooks + sampled profiler + streaming aggregation

The extraction layer is fixed:

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

### 2.0 Fixed `feature_names` schema

`FeatureExtractor.feature_names` returns a fully-determined list of
output keys *before any training step has run*. Three guarantees:

1. **Length-stable across runs.** The list is identical for any
   (architecture, num_layers, sampled-layer strategy, parameter
   groups) tuple regardless of how many epochs the run lasts or
   which validation metrics the user records.
2. **Inspector-driven.** Layer-indexed keys (`L{i}_...`) expand
   against `inspector.get_sampled_layer_indices()` so a 12-layer
   BERT and a 6-layer DistilBERT each get the right per-layer keys.
3. **Self-padding.** `finalize()` fills any column the runtime did
   not emit with `0.0`, so the returned dict's keys equal
   `feature_names` exactly. Downstream classifiers can pin their
   input dimensionality at training time without depending on which
   task / metric / cadence produced a particular row.

The list is built by walking each metric module's
`static_feature_names()` declaration, crossing it with the
windowed-feature suffixes (`_early_mean`, `_mid_slope`, `_final`,
etc.), and unioning in every `val_<raw>` key declared by the task
registry (so CoLA's `val_matthews_correlation_*` and STS-B's
`val_pearson_*` / `val_spearmanr_*` are in the schema even if the
user never runs those tasks).

`compute_window_ranges(total_epochs)` splits the run into thirds —
`early` is the first third, `mid` the middle third, `late` the
last — so the same column names describe a 5-epoch and a 50-epoch
run with semantically comparable definitions. The paper's 10-epoch
schedule maps to `(1-3)`, `(4-6)`, `(7-10)` (a one-epoch shift from
the legacy fixed `(1-3)/(4-7)/(8-10)` mapping; the dataset writer
re-runs the windowed aggregator so historical CSVs may need
regenerating against the new windows).

`MetricCollector.validate_feature_names(expected)` raises
`ValueError` with a missing/unexpected diff when the live schema
diverges from a saved reference — used by
`defaultplusplus.diagnosis.load_pretrained()` to fail closed when
the runtime extractor cannot produce the same columns the
pretrained classifier was trained on.

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
  below; they do not constitute independent schema entries unless
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

The six fixed global gradient statistics are `grad_norm_total`,
`grad_abs_min`, `grad_abs_max`, `grad_zero_ratio`, `gradient_vanish`,
and `gradient_explode`. Update ratios are reported per parameter group
(`update_ratio_{group}`) rather than as a global total, which keeps the
optimization group at 21 features (5 components x 3 plus 6 global). The
earlier `update_ratio_total` key was removed for this reason.

### 2.3 Attention and score signals

Sampled attention-layer patterns are part of the schema. `{layer_idx}`
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
| `L{layer_idx}_pre_softmax_score_mean` / `..._var` / `..._skew` / `..._kurt` | QK score statistics, computed from captured Q/K projection outputs | both | `exact` | `runtime_v1` |
| `L{layer_idx}_qkv_alignment_qk_cos_mean` / `..._qv_cos_mean` / `..._kv_cos_mean` | Direct Q-K, Q-V, K-V head-averaged cosine similarity from post-projection captures | both | `exact` | `runtime_v1` |
| Global aliases: `attention_entropy`, `attention_entropy_mean`, `mass_pad`, `mass_leak`, `cross_example_attention`, `attention_mass_future`, `pre_softmax_score_mean`, `pre_softmax_score_var`, `pre_softmax_score_skew`, `pre_softmax_score_kurt`, `head_similarity_mean`, `head_similarity_max`, `qkv_alignment_qk_cos_mean`, `qkv_alignment_qv_cos_mean`, `qkv_alignment_kv_cos_mean` | Aggregated across sampled layers | both | `exact` | `runtime_v1` |

### 2.4 QKV alignment

| Metric | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `qkv_alignment_qk_cos_mean` / `qkv_alignment_qv_cos_mean` / `qkv_alignment_kv_cos_mean` | Head-averaged cosine similarity between captured post-projection Q, K, V tensors | both | `exact` | `runtime_v1` |

QKV alignment is wired through the sublayer hooks installed by
`SublayerCapture` (`extraction/sublayer_capture.py`). The post-projection
Q, K, V tensors are tapped via forward hooks on the per-layer
projection `nn.Linear` modules. Per-layer cosines are emitted with the
`L{layer_idx}_qkv_alignment_*` prefix and rolled up into the three
global aliases above.

### 2.5 Structural, FFN, residual, LayerNorm behavior

| Metric or pattern | Meaning | Availability | Status | Scope |
|---|---|---|---|---|
| `ffn_delta_l{layer_idx}_mean` | Norm of FFN-sublayer-induced hidden-state delta (FFN output minus FFN input, captured via sublayer hooks) | both | `exact` | `runtime_v1` |
| `residual_cos_l{layer_idx}_mean` | Cosine similarity between FFN-sublayer input and output | both | `exact` | `runtime_v1` |
| `ffn_var_ratio_l{layer_idx}` | FFN output / input variance ratio | both | `exact` | `runtime_v1` |
| `ln_std_l{layer_idx}_mean` | Standard deviation of LayerNorm output, read from the per-layer LN forward hook | both | `exact` | `runtime_v1` |
| `ln_mean_abs_l{layer_idx}_mean` | Mean absolute value of LayerNorm output, read from the per-layer LN forward hook | both | `exact` | `runtime_v1` |
| `ffn_active_dim_frac_l{layer_idx}` | Fraction of active FFN-output dimensions (variance > threshold) | both | `exact` | `runtime_v1` |
| `ffn_out_skew_l{layer_idx}` | FFN-output skewness | both | `exact` | `runtime_v1` |
| Global aggregates: `ffn_delta_mean`, `residual_cos_mean`, `ffn_var_ratio_mean`, `ln_std_mean`, `ln_mean_abs_mean`, `ffn_active_dim_frac_mean`, `ffn_out_skew_mean` | Aggregated across layerwise structural metrics | both | `exact` | `runtime_v1` |
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
| `cache_nll_divergence` | Mean symmetric KL between fresh and cached next-token distributions, sampled at a few positions per probe step | decoder | `exact` | `runtime_v1` |

### 2.9 Benchmark construction interface

The runtime feature schema above is the v1 interface that downstream
diagnostic models pin against. The benchmark construction pipeline
(under `defaultplusplus.benchmark/`, exposed via `defaultpp-benchmark`)
emits two classes of labeled dataset row.

- **Faulty class.** One row per **killed mutant**, from paired clean /
  faulty fine-tunes scored with the exact one-sided sign-flip permutation
  test in `deform.validation.is_killed`. The row carries
  `detection_label = 1` plus the fault category and root-cause labels.
- **Correct class.** One row per retained **label-preserving clean
  variant** of a base model, generated by `deform.generate_clean_variants`
  and run by `deform.run_one_clean_variant`. A variant varies only the
  seed (which reshuffles the training-data order) and hyperparameters
  within behavior-preserving ranges, then is tested against the base
  model with the same kill test. A variant that stays statistically
  indistinguishable from the base model is retained with
  `detection_label = 0` and empty fault labels; one that satisfies the
  killed criterion is discarded. The `--clean-variants N` flag on
  `defaultpp-benchmark` sets how many variants to generate per
  (model, task).

The two classes are symmetric by construction: the faulty path keeps
killed mutants, and the correct path keeps clean variants that are not
killed.

### 2.9.1 Kill-test scoring

The sign-flip test consumes one scalar per (clean, faulty) seed pair.
For each task the scalar is fixed by
[`benchmark.task_metrics.TASK_METRICS`](../defaultplusplus/src/defaultplusplus/benchmark/task_metrics.py)
and follows the standard reporting convention so kill decisions are
comparable to the literature:

| Task          | Arch    | Composite                | `higher_is_better` |
|---------------|---------|--------------------------|--------------------|
| `sst2`        | encoder | accuracy                 | true               |
| `qnli`        | encoder | accuracy                 | true               |
| `rte`         | encoder | accuracy                 | true               |
| `mnli`        | encoder | matched accuracy         | true               |
| `cola`        | encoder | Matthews correlation     | true               |
| `mrpc`        | encoder | (accuracy + F1) / 2      | true               |
| `qqp`         | encoder | (accuracy + F1) / 2      | true               |
| `stsb`        | encoder | (Pearson + Spearman) / 2 | true               |
| `lambada`     | decoder | log-perplexity           | false              |
| `ptb`         | decoder | log-perplexity           | false              |
| `wikitext2`   | decoder | log-perplexity           | false              |
| `openwebtext` | decoder | log-perplexity           | false              |

Adding a new task is a one-entry registration in
`benchmark/task_metrics.py`. The `TaskMetricSpec` declares which raw
metrics the HF Trainer's `compute_metrics` callable must emit and how
they collapse into the kill-test scalar. The runner uses the spec's
`higher_is_better` per configuration, so encoder + decoder tasks can
mix in a single CLI invocation.

The n=5 matched-seed design is fixed by the paper: it is the
smallest n that admits an exact one-sided sign-flip test at α=0.05
(minimum p-value 1/2^5 ≈ 0.031). The runner never aggregates a
partial set of seeds, so the kill-test guarantee is preserved for
every dataset row.

### 2.9.2 Crash isolation and discard logging

A benchmark batch touches dozens of operators per model; some of them
genuinely break the model (NaN logits, OOM, shape mismatch from a
fault that the structural verifier should otherwise catch). The
runner never lets a single configuration take down the whole batch.
Three discard paths are recognized:

| `RunStatus`        | When |
|--------------------|------|
| `verifier_failed`  | the pre-flight `StructuralVerifier` reports `ok=False` (no targets, restoration broken, dynamic wrap leaked outside the intended set) |
| `runtime_error`    | the faulty fine-tune raises an exception on any seed |
| `invalid_metric`   | any seed returns `NaN` or `±Inf` for the test metric |

Discarded configurations carry `mutant=None` and a human-readable
`discard_reason`. The CLI skips them from the output CSV and writes
one JSON record per discard to `<output>.discarded.jsonl` so the
batch leaves an audit trail and the failing operators can be
revisited later. **Clean-run failures bubble up** — they are
environment problems (dataset missing, model wouldn't load) rather
than faults, and the operator running the benchmark needs to see
them.

---

## 3. Scope and future extensions

The v1 surface is complete. The runtime extractor, the diagnostic model,
the benchmark pipeline, and their distribution paths all ship. This
section records what v1 covers and the directions a v2 could extend.

### 3.1 Feature schema

The schema in Section 2 is complete for v1. Every metric is `exact`
except the two noted in the tables. The legacy `attention_score_*`
log-prob proxy was replaced by the exact `pre_softmax_score_*` family
produced via the sublayer hooks.

### 3.2 What v1 ships

- **Single-run anomaly encoding.** `RuntimeNormalizer`
  (`processing/normalizer.py`) loads a learned clean reference and
  converts one live run's feature vector into the shape the diagnostic
  model expects.
- **Pretrained diagnostic-model weights.** The
  `defaultplusplus.diagnosis` API (`load_pretrained`, `Predictor`,
  `save_checkpoint`) and the trained `.pt` checkpoints for the encoder
  and decoder models ship in the wheel under
  `pretrained/weights/{encoder,decoder}.pt`, with matching
  `{encoder,decoder}_reference.npz` runtime references.
- **Benchmark pipeline.** The `defaultpp-benchmark` console script
  ([`benchmark.cli`](../defaultplusplus/src/defaultplusplus/benchmark/cli.py))
  runs paired clean / faulty fine-tunes with crash isolation and the
  per-task metric registry. Per-operator injectors under
  `deform/operator_impls/` cover all 52 catalog entries.
- **Public dataset distribution.** `defaultplusplus.data.download_bench`
  and the `defaultpp-bench-download` console script fetch the benchmark
  CSVs from Zenodo with checksum verification (the CSVs are multiple
  gigabytes and do not ship in the wheel).

### 3.3 Future extensions (beyond v1 scope)

The families below are deliberately out of scope for v1. Supporting them
is a natural direction for future work rather than missing functionality.

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
  output schema.

Changes to this document must be logged in
[`../defaultplusplus/CHANGELOG.md`](../defaultplusplus/CHANGELOG.md).
