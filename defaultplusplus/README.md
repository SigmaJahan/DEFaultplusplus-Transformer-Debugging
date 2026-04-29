# defaultplusplus

Hierarchical fault diagnosis and runtime feature extraction for
HuggingFace transformers.

This document covers the **installable Python package**: API surface,
examples, and the build / publish workflow. For the research-side
reproduction, training, and benchmark-construction flow see
[`RESEARCH.md`](RESEARCH.md). The frozen output schema is in
[`../docs/SPEC.md`](../docs/SPEC.md).

## Install

```bash
pip install defaultplusplus           # core runtime feature extractor
pip install defaultplusplus[hf]       # + HuggingFace Trainer callback
pip install defaultplusplus[viz]      # + matplotlib / seaborn / rich
pip install defaultplusplus[baselines]# + xgboost / imbalanced-learn
pip install defaultplusplus[dev]      # + pytest / build / twine / ruff
pip install defaultplusplus[all]      # hf + viz + baselines
```

Editable install from a local checkout:

```bash
cd defaultplusplus
pip install -e ".[dev,hf]"
```

## Public API

```python
from defaultplusplus import (
    FeatureExtractor,            # manual training loop
    DEFaultPlusCallback,         # HuggingFace Trainer callback
    ExtractionConfig,            # tunable thresholds and sampling cadence
    build_feature_vector,        # TrainingTrace -> fixed-length dict
    build_paired_feature_vector, # paired clean/faulty traces (research only)
)
```

`DEFaultPlusCallback` is resolved lazily so the package itself imports
without the `transformers` extra.

### `FeatureExtractor` lifecycle

```python
fx = FeatureExtractor(model, arch="encoder")  # auto-detects encoder vs decoder
fx.step(loss=..., outputs=..., input_ids=..., attention_mask=...,
        labels=..., optimizer=..., step_time=None)        # once per training step
fx.epoch_end(epoch)                                       # once per epoch
fx.record_validation(epoch, {"accuracy": 0.85, ...})      # at eval checkpoints
features = fx.finalize()                                  # flat dict[str, float]
fx.to_json("features.json")                               # convenience
fx.reset()                                                # discard state for reuse
```

The class is also a context manager:

```python
with FeatureExtractor(model, arch="encoder") as fx:
    ...
# fx.feature_vector is populated on clean exit
```

### Architecture support

| Family       | Examples                                         |
|--------------|--------------------------------------------------|
| **encoder**  | BERT, RoBERTa, DistilBERT, ALBERT-style          |
| **decoder**  | GPT-2, DistilGPT-2, GPT-Neo                      |

`arch` is auto-detected from the model's structure. A wrong hint fails
closed:

```python
FeatureExtractor(bert_model, arch="decoder")
# ValueError: Requested arch='decoder' but the inspector detected 'encoder'.
```

Encoder-decoder architectures (T5, BART) are out of contract for v1.

### Output keys

All output keys are frozen by [`../docs/SPEC.md`](../docs/SPEC.md).
The vector is a flat `dict[str, float]` covering:

- training dynamics: `train_loss`, `train_learning_rate`, `runtime_step_time`, ...
- gradient and update behavior: `grad_norm_total`, `grad_norm_{group}`, `update_ratio_{group}`, ...
- attention and score signals: `L{n}_attention_entropy_mean`, `L{n}_attention_mass_pad_mean`, ...
- structural / FFN / residual / LayerNorm behavior
- logit and task signals: `accuracy`, `f1_score`, `logit_entropy`, `nll`, ...
- positional behavior, cache behavior (decoder)

## Examples

| File | Path | What it shows |
|---|---|---|
| Manual loop | [`examples/extract_during_finetune.py`](examples/extract_during_finetune.py) | DistilBERT + synthetic SST-2 |
| HF Trainer | [`examples/extract_with_hf_trainer.py`](examples/extract_with_hf_trainer.py) | `compute_loss` override + callback |

Run them after installing the `[hf]` extra:

```bash
pip install -e ".[hf]"
python examples/extract_during_finetune.py
python examples/extract_with_hf_trainer.py
```

## Layout

```
defaultplusplus/
  src/defaultplusplus/         the installable package
    __init__.py                FeatureExtractor, DEFaultPlusCallback, etc.
    _version.py                single source of truth for __version__
    api.py                     FeatureExtractor (manual loop)
    hf_callback.py             DEFaultPlusCallback (HF Trainer)
    config.py                  ExtractionConfig dataclass
    py.typed                   PEP 561 marker
    extraction/                metric collection + aggregation
      inspector.py             auto-discovers transformer structure
      collector.py             orchestrates per-step metric modules
      aggregator.py            Welford running statistics
      feature_construction.py  layer / step / epoch / phase aggregation
      metrics/                 attention, gradient, logit, structural, ...
    deform/                    mutation engine (research)
      operators.py             45 mutation operators
      injection.py             StaticFault / DynamicFault context managers
      validation.py            structural verifier + sign-flip kill test
      fault_config.py          FaultConfiguration / Mutant types
    benchmark/                 benchmark construction (research)
      config_grid.py
      runner.py
      dataset_writer.py
    diagnosis/                 reserved for runtime diagnostic model
    pretrained/                reserved for shipped checkpoints
    processing/                reserved for runtime feature processor
    ui/                        reserved for runtime UI helpers

  hierarchical_graph_category_rootcause/  research-side training drivers
                                          (see RESEARCH.md)
  configs/base.yaml                       hyperparameter config
  examples/                               runnable demos
  scripts/                                local + Compute Canada scripts
  tests/                                  pytest suite (58 tests)
  pyproject.toml                          PEP 621 metadata + build config
  LICENSE                                 Apache-2.0
  CHANGELOG.md                            version history
  README.md                               this file
  RESEARCH.md                             research-side runbook
```

## Build and publish

The package follows the standard PEP 517 / PEP 621 layout. Building
and uploading uses `build` + `twine`.

### Bump the version

Update [`src/defaultplusplus/_version.py`](src/defaultplusplus/_version.py)
following Semantic Versioning (`MAJOR.MINOR.PATCH`). Add an entry at
the top of [`CHANGELOG.md`](CHANGELOG.md). The version field in
`pyproject.toml` is dynamic and reads from `_version.py` so the two
cannot drift apart.

### Build locally

```bash
bash scripts/build_pypi.sh
```

Cleans `dist/`, runs `python -m build` (sdist + wheel), and validates
with `twine check`.

### Publish to TestPyPI

```bash
bash scripts/build_pypi.sh --testpypi
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            defaultplusplus
```

Authentication uses the standard `twine` config: `~/.pypirc` or
`TWINE_USERNAME=__token__` + `TWINE_PASSWORD=<api-token>`.

### Publish to PyPI

```bash
bash scripts/build_pypi.sh --pypi
pip install --upgrade defaultplusplus
```

A version on PyPI is permanent; bump before re-uploading.

## Tests

```bash
pytest tests/                # full suite
pytest tests/test_dry_run.py # smoke suite (~1s, no real models)
pytest tests/test_feature_extractor.py  # uses real DistilBERT/GPT-2 (~15s)
bash scripts/dry_run_local.sh --quick   # imports + smoke + FPG sanity
```

## Dependencies

Required at runtime:

- Python ≥ 3.10
- `torch` ≥ 2.1
- `transformers` ≥ 4.30
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml`, `tqdm`, `joblib`

Optional extras: `hf` (accelerate, datasets), `viz` (matplotlib, seaborn,
rich), `baselines` (xgboost, imbalanced-learn), `dev` (pytest, build,
twine, ruff).

## Versioning

Semantic Versioning 2.0:

- **MAJOR** for incompatible API changes (removing a public symbol,
  changing a function signature in a non-additive way).
- **MINOR** for backwards-compatible additions.
- **PATCH** for backwards-compatible bug fixes.

The current version is read from
[`src/defaultplusplus/_version.py`](src/defaultplusplus/_version.py)
and surfaced as `defaultplusplus.__version__`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
