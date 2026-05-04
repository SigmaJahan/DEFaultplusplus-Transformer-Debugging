# defaultplusplus

[![CI](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/actions/workflows/ci.yml)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20019817.svg)](https://doi.org/10.5281/zenodo.20019817)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20018623.svg)](https://doi.org/10.5281/zenodo.20018623)

Hierarchical fault diagnosis and runtime feature extraction for
HuggingFace transformers.

This document covers the **installable Python package**: API surface,
examples, and the build / publish workflow. For the research-side
reproduction, training, and benchmark-construction flow see
[`RESEARCH.md`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/RESEARCH.md). The frozen output schema is in
[`docs/SPEC.md`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/docs/SPEC.md).

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

from defaultplusplus.diagnosis import (
    load_pretrained,             # arch -> Predictor (raises if no weights)
    Predictor,                   # .predict(features) -> Diagnosis
    Diagnosis,                   # 3-level result dataclass
)
```

`DEFaultPlusCallback` is resolved lazily so the package itself imports
without the `transformers` extra.

### Three-level diagnosis

```python
from defaultplusplus import FeatureExtractor
from defaultplusplus.diagnosis import load_pretrained

with FeatureExtractor(model, arch="encoder") as fx:
    # ... your training loop calls fx.step(...) and fx.epoch_end(...) ...
    features = fx.finalize()

predictor = load_pretrained("encoder")            # ships in the wheel
diagnosis = predictor.predict(features)
print(diagnosis.to_dict())
# {
#   'is_faulty':       True,
#   'detection_prob':  0.92,
#   'category':        'qkv',
#   'category_prob':   0.81,
#   'root_cause':      'zero_query',
#   'root_cause_prob': 0.74,
#   'group_importance': {'qkv_alignment': 3.2, 'attention': 1.7, ...},
# }
```

The `Predictor` validates the live `feature_names` schema against the
one bundled in the checkpoint, so a model trained against schema X
refuses to score features built against schema Y.

Pretrained weights ship inside the wheel under
`defaultplusplus/pretrained/weights/{encoder,decoder}.pt`, with
matching `{encoder,decoder}_reference.npz` runtime references for
`RuntimeNormalizer`. Train your own from the public benchmark with
`scripts/train_diagnoser.py` if you want to reproduce or fine-tune.

### Single-run anomaly encoding

The diagnostic model was trained against a paper-aligned feature
schema. At runtime, `RuntimeNormalizer` turns a live extractor's dict
into the exact schema the model expects:

```python
from defaultplusplus.processing import RuntimeNormalizer
from defaultplusplus.diagnosis import load_pretrained

norm = RuntimeNormalizer.load("encoder")
predictor = load_pretrained("encoder")
encoded = norm.encode(features, mode="raw")    # fills missing keys with baseline median
diagnosis = predictor.predict(encoded)
```

Pass `mode="anomaly"` for `(value − median) / mad` z-scores instead.

### Visualization

The `[viz]` extra adds a self-contained HTML report writer plus seven
matplotlib plot functions:

```python
from defaultplusplus.viz import save_diagnosis_report
save_diagnosis_report(diagnosis, encoded, "run.html")  # standalone HTML
```

Individual plots (`plot_diagnosis`, `plot_group_importance`,
`plot_per_layer_heatmap`, `plot_training_trace`,
`plot_attention_pattern`, `plot_qkv_alignment`,
`plot_feature_anomaly`) return `matplotlib.Figure` objects.

### Public benchmark dataset

The training data CSVs (~360 MB) live on Zenodo, not in the wheel.
Fetch them on demand:

```bash
defaultpp-bench-download                        # ~/.cache/defaultplusplus/bench/v1
```

Or programmatically:

```python
from defaultplusplus.data import download_bench
path = download_bench(version="v1")             # checksum-verified, idempotent
```

DOI: [10.5281/zenodo.20018623](https://doi.org/10.5281/zenodo.20018623).

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

Encoder-decoder architectures (T5, BART) are out of scope for v1.

### Supported benchmark tasks

The kill-test scoring is registry-driven (one `TaskMetricSpec` per
task). The composite per task follows the standard GLUE / LM
reporting convention so kill decisions are comparable to the
literature:

| Task           | Arch     | Composite                         | Direction |
|----------------|----------|-----------------------------------|-----------|
| `sst2`         | encoder  | accuracy                          | ↑         |
| `qnli`         | encoder  | accuracy                          | ↑         |
| `rte`          | encoder  | accuracy                          | ↑         |
| `mnli`         | encoder  | accuracy (matched validation)     | ↑         |
| `cola`         | encoder  | Matthews correlation              | ↑         |
| `mrpc`         | encoder  | (accuracy + F1) / 2               | ↑         |
| `qqp`          | encoder  | (accuracy + F1) / 2               | ↑         |
| `stsb`         | encoder  | (Pearson + Spearman) / 2          | ↑         |
| `wikitext2`    | decoder  | eval_loss                         | ↓         |

Adding a new task means registering one `TaskMetricSpec` in
[`benchmark/task_metrics.py`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/src/defaultplusplus/benchmark/task_metrics.py);
nothing else in the runner / CLI changes.

### Output keys

All output keys are frozen by [`docs/SPEC.md`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/docs/SPEC.md).
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
| Manual loop | [`examples/extract_during_finetune.py`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/examples/extract_during_finetune.py) | DistilBERT + synthetic SST-2 |
| HF Trainer | [`examples/extract_with_hf_trainer.py`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/examples/extract_with_hf_trainer.py) | `compute_loss` override + callback |

Run them after installing the `[hf]` extra:

```bash
pip install -e ".[hf]"
python examples/extract_during_finetune.py
python examples/extract_with_hf_trainer.py
```

## Benchmark CLI

Once `[hf]` is installed, `defaultpp-benchmark` runs a paired
clean / faulty fine-tune for every (model × task × operator ×
severity × seed-tuple) configuration and writes one CSV row per
killed mutant. The kill test uses the per-task scalar from the
metric registry (see Supported benchmark tasks above).

```bash
defaultpp-benchmark \
  --arch encoder \
  --models bert-base-uncased \
  --tasks sst2 \
  --operators QZQ,FCA \
  --severities low \
  --seeds 42,123,456,789,101112 \
  --output data/encoder_benchmark.csv
```

A configuration is **discarded** (not crashed) when:

- the pre-flight `StructuralVerifier` reports the injector targets no
  parameters or fails to restore on exit,
- the faulty fine-tune raises an exception on any seed, or
- any seed returns a non-finite metric (NaN / ±Inf).

Discarded configs are skipped from the CSV and recorded in
`<output>.discarded.jsonl` (one record per line) so they can be
revisited later. The CLI prints a summary at the end:

```
[defaultpp-benchmark] wrote 124/130 row(s) to data/encoder_benchmark.csv; 6 discarded
[defaultpp-benchmark] discard summary: runtime_error=4, verifier_failed=2
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
      sublayer_capture.py      forward hooks for FFN/LN/Q/K/V taps
      feature_construction.py  layer / step / epoch / phase aggregation
      metrics/                 attention, gradient, logit, structural, ...
    deform/                    mutation engine (research)
      operators.py             45 mutation operators (catalog)
      operator_impls/          per-operator injector implementations
      injection.py             StaticFault / DynamicFault context managers
      validation.py            structural verifier + sign-flip kill test
      fault_config.py          FaultConfiguration / Mutant types
    benchmark/                 benchmark construction (research)
      cli.py                   defaultpp-benchmark entry point
      config_grid.py
      runner.py                paired runs + crash isolation
      task_metrics.py          per-task kill-test metric registry
      dataset_writer.py
    diagnosis/                 runtime Predictor + load_pretrained()
      model.py                 HierarchicalDiagnosisModel
      _group_encoder.py        GroupEncoder + GraphAggregator
      predictor.py             Predictor, save_checkpoint, load_pretrained
    processing/                FeatureProcessor + RuntimeNormalizer
      feature_processor.py     6-step preprocessing pipeline
      feature_groups.py        feature → group routing
      normalizer.py            single-run anomaly encoding
    pretrained/                shipped diagnostic-model checkpoints
      weights/encoder.pt       trained encoder diagnoser
      weights/decoder.pt       trained decoder diagnoser
      weights/*.npz            RuntimeNormalizer baseline references
    data/                      benchmark download (defaultpp-bench-download)
    viz/                       matplotlib plots + HTML report writers
    ui/                        reserved for future CLI helpers

  hierarchical_graph_category_rootcause/  research-side training drivers
                                          (see RESEARCH.md)
  configs/base.yaml                       hyperparameter config
  examples/                               runnable demos
  scripts/                                local + Compute Canada scripts
  tests/                                  pytest suite (201 tests)
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

Update [`src/defaultplusplus/_version.py`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/src/defaultplusplus/_version.py)
following Semantic Versioning (`MAJOR.MINOR.PATCH`). Add an entry at
the top of [`CHANGELOG.md`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/CHANGELOG.md). The version field in
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
[`src/defaultplusplus/_version.py`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/src/defaultplusplus/_version.py)
and surfaced as `defaultplusplus.__version__`.

## License

Apache-2.0. See [`LICENSE`](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/blob/main/defaultplusplus/LICENSE).
