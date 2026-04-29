# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- DEForm mutation engine: 45 mutation operators across 12 transformer
  components, static + dynamic injection context managers, structural
  verifier, and exact one-sided sign-flip permutation test.
- Benchmark construction pipeline: configuration-grid enumeration,
  per-configuration runner, and CSV / Parquet shard writer.
- Feature-construction pipeline: layer / step / epoch / training-phase
  aggregation that produces the fixed-length feature vector consumed
  by the diagnostic model.
- Compute Canada SLURM scripts under `scripts/cc/`.
- Local end-to-end dry-run harness (`scripts/dry_run_local.sh` and
  `tests/test_dry_run.py`).

### Changed
- Feature-group taxonomy renamed to match the diagnostic model's
  twelve-encoder / thirteen-decoder schema (`qkv_alignment`,
  `ffn_output`, `residual_stream`, `output`, `cache`,
  `representation_drift`, `validation_perf`).
- Hierarchical loss formula reorganized into the canonical form
  `L = L_detect + alpha * L_cat + lambda_rc * L_rc + L_sep` with
  `L_sep = beta * L_ctr + gamma * L_pm`.
- Graph aggregator implements the canonical message-passing update
  `H = ReLU(A_hat * H * W_msg)` with row-normalized adjacency, three
  rounds, and a learnable matrix per round.
- Root-cause explanation reports per-group importance from the
  predicted vs. nearest-alternative prototype margin.
- Default training hyperparameters: 150 epochs, three message-passing
  rounds, early-stopping patience of 20, gamma of 0.3.

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
