# DEFault++

[![CI](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/actions/workflows/ci.yml)

Hierarchical fault diagnosis and runtime feature extraction for
HuggingFace transformers.

DEFault++ inspects a fine-tuning run and answers three questions:

  1. **Detection** – is something wrong with this run?
  2. **Categorization** – which transformer subsystem is responsible
     (QKV, masking, LayerNorm, …)?
  3. **Root cause** – what specific bug pattern fits the evidence
     (e.g. `zero_query`, `mask_application`, `weight_scaling`)?

The library exposes the runtime feature extractor as a clean public
API; the diagnostic model is shipped separately on the research side.

## Install

```bash
pip install defaultplusplus
```

For the HuggingFace `Trainer` integration:

```bash
pip install defaultplusplus[hf]
```

## Quick start

### Manual training loop

```python
import time
from defaultplusplus import FeatureExtractor

with FeatureExtractor(model, arch="encoder") as fx:
    for epoch in range(num_epochs):
        for step, batch in enumerate(loader):
            t0 = time.perf_counter()
            outputs = model(**batch,
                            output_attentions=True,
                            output_hidden_states=True)
            outputs.loss.backward()
            optimizer.step(); optimizer.zero_grad()
            fx.step(loss=outputs.loss, outputs=outputs,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    optimizer=optimizer,
                    step_time=time.perf_counter() - t0)
        fx.epoch_end(epoch)
        fx.record_validation(epoch, eval_loop(model))

feature_vector = fx.feature_vector  # populated on context-manager exit
```

### HuggingFace `Trainer`

```python
from defaultplusplus.hf_callback import DEFaultPlusCallback

trainer = Trainer(
    model=model, args=args,
    callbacks=[DEFaultPlusCallback(out_path="features.json", arch="encoder")],
)
trainer.train()
```

The callback enables `output_attentions` / `output_hidden_states` on
the model's config automatically and writes the finalized feature
vector to `out_path` when training ends.

To capture per-step attention weights and hidden states (for the
attention-internal metrics), wrap the trainer's `compute_loss` to
forward inputs and outputs into the callback:

```python
class _Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        callback.capture_inputs(dict(inputs))
        outputs = model(**inputs)
        callback.capture_outputs(outputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
```

### Output

The finalized `feature_vector` is a flat `dict[str, float]` keyed by
the schema in `docs/SPEC.md`. Save it as JSON, feed it to the
diagnostic model, or use it as input to your own classifier.

## Public API

| Symbol | Purpose |
|---|---|
| `FeatureExtractor` | manual-loop feature extractor with `step()`, `epoch_end()`, `record_validation()`, `finalize()`, `to_json()` |
| `DEFaultPlusCallback` | drop-in HuggingFace `TrainerCallback` |
| `ExtractionConfig` | thresholds, sampling cadence, special-token IDs |
| `build_feature_vector` | turn a typed `TrainingTrace` into the fixed-length feature vector |
| `build_paired_feature_vector` | paired clean / faulty traces (used during benchmark construction) |

Documented endpoints, supported model families, and the frozen output
schema live in [`docs/SPEC.md`](docs/SPEC.md).

## Examples

- [`defaultplusplus/examples/extract_during_finetune.py`](defaultplusplus/examples/extract_during_finetune.py) — manual loop
- [`defaultplusplus/examples/extract_with_hf_trainer.py`](defaultplusplus/examples/extract_with_hf_trainer.py) — HF `Trainer`

## Repository layout

```
DEFaultplusplus-Transformer-Debugging/
  defaultplusplus/         installable Python package + research drivers
  data/                    DEFault-bench CSVs
  baselines/               baseline detection scripts
  realworld_benchmark/     real-world GitHub-issue evaluation
  user_study/              developer-study assets
  docs/                    output schema + roadmap (SPEC.md)
  DEFault++.pdf            scientific reference
  README.md                this file (user-facing)
```

## Documentation

- [`README.md`](README.md) — this file. PyPI install, quick start,
  public API.
- [`defaultplusplus/README.md`](defaultplusplus/README.md) — package
  reference: full API, examples, build / publish workflow.
- [`docs/SPEC.md`](docs/SPEC.md) — output schema, architectural
  principles, and roadmap.
- [`DEFault++.pdf`](DEFault++.pdf) — scientific reference.

The research-side reproduction guide and the developer notebook for
the diagnostic model itself live in
[`defaultplusplus/RESEARCH.md`](defaultplusplus/RESEARCH.md).

## License

Apache-2.0. See [`defaultplusplus/LICENSE`](defaultplusplus/LICENSE).

## Citation

DEFault++ is unpublished research. If you use this code, please cite
the repository directly:

```bibtex
@software{defaultplusplus,
  title  = {{DEFault++}: Hierarchical Fault Diagnosis and Runtime Feature
            Extraction for HuggingFace Transformers},
  author = {Jahan, Sigma},
  year   = {2026},
  url    = {https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging},
  version = {0.2.0},
  note   = {Software repository; manuscript in preparation.}
}
```
