# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `defaultplusplus.diagnosis` runtime API: `load_pretrained(arch)` →
  `Predictor.predict(features)` → `Diagnosis(is_faulty, category,
  root_cause, group_importance, ...)`. The schema the model was
  trained against is bundled inside the checkpoint and validated
  against the live `FeatureExtractor.feature_names` at load time, so
  a model trained on one schema cannot silently consume features
  from another. `PretrainedWeightsMissingError` raised with a clear
  message when no `.pt` is on disk.
- `scripts/train_diagnoser.py` training driver. Reads the paper-aligned
  benchmark CSV (`--csv`) or synthesizes labels from random data
  (`--synthetic`, for development), trains a
  `HierarchicalDiagnosisModel`, and writes a v1 checkpoint via
  `defaultplusplus.diagnosis.save_checkpoint`.
- Checkpoint format v1 bundles: `feature_names`, `category_names`,
  `category_sizes`, `rootcause_names`, `group_names`, `model_state_dict`,
  scaler `mean` / `scale`, per-category prototype tensors, and the
  `model_kwargs` needed to reconstruct the model class. Format
  version is checked at load time; future bumps just add an `if`.
- DEForm mutation engine: 45 mutation operators across 12 transformer
  components, static + dynamic injection context managers, structural
  verifier, and exact one-sided sign-flip permutation test.
- Per-operator implementations for all 45 catalog entries under
  `deform/operator_impls/` so the benchmark runner can resolve any
  operator by ID without a custom `injector_factory` callable.
- KV-cache mutation operators (`CST`, `COB`, `CTR`, `CLK`) now mutate
  the live `DynamicCache` (or legacy tuple) — truncate / shift / serve
  one-step-stale snapshots / cross-request leak — and demonstrably
  move the new `cache_nll_divergence` metric.
- `extraction.sublayer_capture.SublayerCapture`: forward hooks on each
  layer's attention, FFN, LayerNorm submodules plus the Q/K/V
  projection `Linear`s. Promotes `ffn_delta_*`, `residual_cos_*`,
  `ffn_var_ratio_*`, `ln_std_*`, `ln_mean_abs_*`,
  `ffn_active_dim_frac_*`, `ffn_out_skew_*` from `reconstructed` to
  `exact`, and emits new `qkv_alignment_qk_cos_mean / _qv_cos_mean /
  _kv_cos_mean` direct cosines.
- `cache_nll_divergence` (decoder only): mean symmetric KL between
  fresh and cached next-token distributions, sampled at a few
  positions per probe step. Promoted from `not_available` to `exact`.
- `defaultpp-benchmark` console script and end-to-end CLI driver
  (`benchmark.cli`) that produces `data/*.csv` from scratch via HF
  Trainer for any combination of supported models / tasks / operators.
- Crash isolation in the runner: `RunStatus` enum + `RunOutcome.status`
  + `discard_reason`. Verifier failures, faulty-run exceptions, and
  non-finite test metrics each discard the configuration without
  affecting other runs in the batch. Discards are written to a
  `*.discarded.jsonl` log next to the dataset CSV.
- Per-task metric registry (`benchmark.task_metrics.TASK_METRICS`)
  defining the scalar that feeds the kill test for each supported
  task: SST-2 / QNLI / RTE / MNLI / CoLA use single metrics, MRPC /
  QQP use the GLUE `(accuracy + F1) / 2` composite, STS-B uses
  `(Pearson + Spearman) / 2`, WikiText uses eval loss.
- Benchmark construction pipeline: configuration-grid enumeration,
  per-configuration runner, and CSV / Parquet shard writer.
- Feature-construction pipeline: layer / step / epoch / training-phase
  aggregation that produces the fixed-length feature vector consumed
  by the diagnostic model. Equation 7.19 dimensions pinned to 1600
  (encoder) / 1705 (decoder).
- Compute Canada SLURM scripts under `scripts/cc/`.
- Local end-to-end dry-run harness (`scripts/dry_run_local.sh` and
  `tests/test_dry_run.py`).

### Changed
- `StructuralVerifier.verify_static` now rejects silent no-op faults:
  if `expected_param_names` is non-empty but no parameter actually
  changed, the verifier fails. Bound-method comparison fixed via the
  `_callable_identity` helper so dynamic verification compares
  `(__self__, __func__)` rather than fresh bound-method objects.
- `QSW` operator rewritten to swap `query.weight ↔ key.weight` (and
  biases) within each attention block. The previous adjacent-pair
  positional swap silently no-op'd on standard HF models.
- Cache operators (`CST`, `COB`, `CTR`, `CLK`) wrap `model.forward`
  instead of per-layer attention so they see the whole `DynamicCache`
  once per forward and can mutate per-layer slices selectively.
- Feature-group taxonomy renamed to match the diagnostic model's
  twelve-encoder / thirteen-decoder schema (`qkv_alignment`,
  `ffn_output`, `residual_stream`, `output`, `cache`,
  `representation_drift`, `validation_perf`).
- Hierarchical loss formula reorganized into the form
  `L = L_detect + alpha * L_cat + lambda_rc * L_rc + L_sep` with
  `L_sep = beta * L_ctr + gamma * L_pm`.
- Graph aggregator implements the message-passing update
  `H = ReLU(A_hat * H * W_msg)` with row-normalized adjacency, three
  rounds, and a learnable matrix per round.
- Root-cause explanation reports per-group importance from the
  predicted vs. nearest-alternative prototype margin.
- Default training hyperparameters: 150 epochs, three message-passing
  rounds, early-stopping patience of 20, gamma of 0.3.

### Removed
- Legacy `L{layer_idx}_attention_score_var` / `..._score_skew` keys
  (log-prob proxy on attention probabilities). The exact
  `pre_softmax_score_*` family (computed from captured Q/K via the
  sublayer hooks) is the single score-shape signal. **MAJOR bump.**

### Fixed
- `_compute_pre_softmax_stats` now reads captured Q/K from the
  sublayer hooks instead of recomputing the projections on the layer
  input (the recomputation drifted under operators that wrap
  attention preprocessing).
- Runner no longer aggregates a partial set of seeds when one seed
  crashes: any per-seed exception or non-finite metric discards the
  whole configuration so the n=5 kill-test guarantee is preserved.

## [0.2.0] - 2026-04-29

### Added
- Public feature-extraction API: `FeatureExtractor` (manual training
  loop) and `DEFaultPlusCallback` (HuggingFace `Trainer` callback).
- HF `Trainer` integration verified end-to-end with real DistilBERT
  and GPT-2 model checkpoints.
- `extraction.feature_construction` aggregator that converts collector
  output into the fixed-length diagnostic-model feature vector.
- Apache-2.0 LICENSE, `MANIFEST.in`, `CHANGELOG.md`, `py.typed` marker,
  and PyPI-ready `pyproject.toml` (PEP 621 metadata, dynamic version,
  trove classifiers, project URLs).
- Build / publish workflow under `scripts/build_pypi.sh`.

### Fixed
- KV-cache metric module handles modern HuggingFace `DynamicCache`
  objects in addition to the legacy tuple-of-tuples shape.
- Feature-construction band-index helper no longer indexes past the
  end of the array when the run has fewer than three epochs / steps.

## [0.1.0]

### Added
- Initial research artifact: hierarchical fault-diagnosis model,
  ablation drivers, baseline comparisons, and the
  `data/` mutation-dataset loader.
