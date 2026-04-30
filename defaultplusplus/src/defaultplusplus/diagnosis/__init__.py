"""Runtime diagnostic-model API.

Public surface:

    from defaultplusplus.diagnosis import load_pretrained, Predictor

    predictor = load_pretrained("encoder")          # raises if no weights
    diagnosis = predictor.predict(feature_vector)   # 3-level dict

The pretrained checkpoint files live under
``defaultplusplus/pretrained/weights/{arch}.pt`` and are produced by
``scripts/train_diagnoser.py``. They are *not* shipped in the wheel —
download them via ``defaultpp-bench-download`` (or train your own
with the script).

The schema the checkpoint was trained against is bundled inside the
``.pt`` file as ``feature_names`` and validated against the live
``FeatureExtractor.feature_names`` at load time, so a model trained
on v0.3.0 schema cannot silently consume features from a different
version.
"""
from __future__ import annotations

from .predictor import (
    Diagnosis,
    Predictor,
    PretrainedWeightsMissingError,
    load_pretrained,
    save_checkpoint,
    weights_path,
)

__all__ = [
    "Diagnosis",
    "Predictor",
    "PretrainedWeightsMissingError",
    "load_pretrained",
    "save_checkpoint",
    "weights_path",
]
