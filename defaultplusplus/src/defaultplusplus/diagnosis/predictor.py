"""Pretrained diagnostic-model loader and inference wrapper.

This module is the user-facing path from a feature dictionary to a
three-level fault diagnosis. Two public entry points:

    load_pretrained(arch)       returns a ``Predictor`` ready to call
                                 ``.predict(features)``.
    save_checkpoint(state, ...) writes the checkpoint format the
                                 training driver produces and that
                                 ``load_pretrained`` consumes.

The checkpoint format (``format_version=1``) bundles everything
``Predictor`` needs at inference time: the model's state dict, the
input scaler statistics, the per-category prototype tensors used for
the explanation step, the schema (``feature_names``) the model was
trained against, and category / root-cause label vocabularies. The
schema is validated against the runtime extractor's
``feature_names`` at load time so a model trained on one version
cannot silently consume features from another.

Pretrained weights are not shipped in the wheel; the package looks
for them in ``defaultplusplus/pretrained/weights/{arch}.pt``. The
training driver in ``scripts/train_diagnoser.py`` produces them; a
future ``defaultpp-bench-download`` console script will fetch the
released versions from a public mirror.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .._version import __version__


# Pretrained checkpoints live under ``pretrained/weights/`` so the
# import path is stable: ``defaultplusplus/pretrained/weights/<arch>.pt``.
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "pretrained" / "weights"

CHECKPOINT_FORMAT_VERSION = "1"


class PretrainedWeightsMissingError(FileNotFoundError):
    """Raised when ``load_pretrained(arch)`` cannot find weights on disk.

    The error message names the expected path and points the caller at
    the training driver so they can produce weights themselves.
    """


@dataclass(frozen=True)
class Diagnosis:
    """One full diagnosis produced by :meth:`Predictor.predict`.

    Attributes:
        is_faulty:       Stage 1. ``True`` if the run looks faulty.
        detection_prob:  P(faulty) from the detection head.
        category:        Stage 2. Category name (e.g. ``"qkv"``,
                         ``"masking"``). ``None`` when ``not is_faulty``.
        category_prob:   P(category) for the predicted category.
        root_cause:      Stage 3. Root-cause label inside the category.
                         ``None`` when ``not is_faulty`` or the
                         category has only one root cause.
        root_cause_prob: P(root_cause) from the per-category head.
        group_importance: dict mapping feature-group name to the
                          per-group margin between the predicted and
                          nearest-alternative prototype. Higher values
                          support the prediction more strongly. Empty
                          dict when stage 3 did not run.
    """
    is_faulty: bool
    detection_prob: float
    category: Optional[str] = None
    category_prob: float = 0.0
    root_cause: Optional[str] = None
    root_cause_prob: float = 0.0
    group_importance: Mapping[str, float] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_faulty": bool(self.is_faulty),
            "detection_prob": float(self.detection_prob),
            "category": self.category,
            "category_prob": float(self.category_prob),
            "root_cause": self.root_cause,
            "root_cause_prob": float(self.root_cause_prob),
            "group_importance": dict(self.group_importance or {}),
        }


def weights_path(arch: str) -> Path:
    """Return the on-disk location for ``{arch}.pt`` weights."""
    if arch not in ("encoder", "decoder"):
        raise ValueError(
            f"unknown arch {arch!r}; expected 'encoder' or 'decoder'"
        )
    return _WEIGHTS_DIR / f"{arch}.pt"


def save_checkpoint(
    *,
    path: Path | str,
    arch: str,
    feature_names: Sequence[str],
    category_names: Sequence[str],
    category_sizes: Mapping[str, int],
    rootcause_names: Mapping[str, Sequence[str]],
    group_names: Sequence[str],
    model_state_dict: Mapping[str, Any],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    prototypes: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a v1 checkpoint to ``path`` and return the resolved path.

    The format is a plain ``torch.save`` of a Python dict. Keeping the
    format simple lets us version it cheaply: a future format_version
    bump (e.g. when adding a new label level) is one ``if`` in
    ``load_pretrained``.
    """
    import torch

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "package_version": __version__,
        "arch": arch,
        "feature_names": list(feature_names),
        "category_names": list(category_names),
        "category_sizes": dict(category_sizes),
        "rootcause_names": {k: list(v) for k, v in rootcause_names.items()},
        "group_names": list(group_names),
        "model_state_dict": dict(model_state_dict),
        "scaler_mean": np.asarray(scaler_mean, dtype=np.float64),
        "scaler_scale": np.asarray(scaler_scale, dtype=np.float64),
        "prototypes": dict(prototypes),
        "model_kwargs": dict(model_kwargs),
    }
    if extra:
        payload["extra"] = dict(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def load_pretrained(arch: str, *, weights: Path | str | None = None,
                    strict_schema: bool = True) -> "Predictor":
    """Load a :class:`Predictor` for ``arch``.

    Args:
        arch:           ``"encoder"`` or ``"decoder"``.
        weights:        optional explicit path to a ``.pt`` file. When
                        omitted we look under
                        ``defaultplusplus/pretrained/weights/{arch}.pt``.
        strict_schema:  when ``True`` (default), a Predictor refuses to
                        score a feature dictionary whose keys differ
                        from the schema baked into the checkpoint. Set
                        to ``False`` only when you know the consumer
                        will subset the columns themselves.

    Raises:
        :class:`PretrainedWeightsMissingError` if the weights file
        is not found, with a message that names the expected path
        and the training-driver script to produce one.
    """
    path = Path(weights) if weights is not None else weights_path(arch)
    if not path.exists():
        raise PretrainedWeightsMissingError(
            f"No pretrained weights at {path}. Either:\n"
            f"  1. Run scripts/train_diagnoser.py --arch {arch} --output {path} "
            f"to train your own, or\n"
            f"  2. Wait for the v1 release blob and download via "
            f"``defaultpp-bench-download``."
        )
    return Predictor.from_checkpoint(path, strict_schema=strict_schema)


class Predictor:
    """Inference wrapper around a trained ``HierarchicalDiagnosisModel``.

    Construct via :func:`load_pretrained`. Use ``.predict(features)``
    to get a :class:`Diagnosis` for a single feature dictionary
    produced by ``FeatureExtractor.finalize()``.
    """

    def __init__(
        self,
        *,
        arch: str,
        feature_names: Sequence[str],
        category_names: Sequence[str],
        category_sizes: Mapping[str, int],
        rootcause_names: Mapping[str, Sequence[str]],
        group_names: Sequence[str],
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        model: Any,
        prototypes: Mapping[str, Any] | None = None,
        strict_schema: bool = True,
    ) -> None:
        self.arch = arch
        self.feature_names = list(feature_names)
        self._feature_index = {name: i for i, name in enumerate(self.feature_names)}
        self.category_names = list(category_names)
        self.category_sizes = dict(category_sizes)
        self.rootcause_names = {k: list(v) for k, v in rootcause_names.items()}
        self.group_names = list(group_names)
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        # Avoid divide-by-zero on constant columns.
        self.scaler_scale = np.where(
            self.scaler_scale > 1e-12, self.scaler_scale, 1.0,
        )
        self.model = model
        self.strict_schema = strict_schema
        # Restore prototype tensors onto the model so ``diagnose_proto``
        # works post-load.
        if prototypes:
            for cat_name, proto in prototypes.items():
                self.model._prototypes[cat_name] = proto
        self.model.eval()

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def from_checkpoint(cls, path: Path | str, *, strict_schema: bool = True) -> "Predictor":
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        format_version = payload.get("format_version")
        if format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint {path} has format_version={format_version!r}; "
                f"this version of defaultplusplus understands "
                f"{CHECKPOINT_FORMAT_VERSION!r}"
            )

        arch = payload["arch"]
        model = _build_model_from_kwargs(payload["model_kwargs"])
        model.load_state_dict(payload["model_state_dict"])

        return cls(
            arch=arch,
            feature_names=payload["feature_names"],
            category_names=payload["category_names"],
            category_sizes=payload["category_sizes"],
            rootcause_names=payload["rootcause_names"],
            group_names=payload["group_names"],
            scaler_mean=payload["scaler_mean"],
            scaler_scale=payload["scaler_scale"],
            model=model,
            prototypes=payload.get("prototypes") or {},
            strict_schema=strict_schema,
        )

    # ── Inference ───────────────────────────────────────────────────
    def predict(self, features: Mapping[str, float]) -> Diagnosis:
        """Run the three-level diagnosis on one feature dictionary.

        ``features`` must be the dict returned by
        ``FeatureExtractor.finalize()``. With ``strict_schema=True``
        the keys must match what the model was trained on; missing
        keys raise ``ValueError`` (use ``strict_schema=False`` to fill
        missing columns with 0.0 silently).
        """
        x = self._vectorize(features)
        return self._predict_single(x)

    def predict_batch(self, batch: Sequence[Mapping[str, float]]) -> list[Diagnosis]:
        """Vectorized version of :meth:`predict`."""
        if not batch:
            return []
        return [self._predict_single(self._vectorize(f)) for f in batch]

    def validate_feature_names(self, expected: Sequence[str]) -> None:
        """Raise if ``expected`` doesn't match the bundled schema.

        Mirrors :meth:`MetricCollector.validate_feature_names`. Useful
        for asserting a pipeline's runtime extractor agrees with the
        checkpoint *before* any predictions happen.
        """
        live_set = set(self.feature_names)
        expected_set = set(expected)
        missing = sorted(expected_set - live_set)
        unexpected = sorted(live_set - expected_set)
        if not missing and not unexpected:
            return
        parts = []
        if missing:
            parts.append(
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
        if unexpected:
            parts.append(
                f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
            )
        raise ValueError(
            "feature_names schema mismatch (predictor vs expected): "
            + "; ".join(parts)
            + f" — predictor={len(self.feature_names)}, expected={len(expected)}."
        )

    # ── Internals ───────────────────────────────────────────────────
    def _vectorize(self, features: Mapping[str, float]) -> np.ndarray:
        """Build a ``(input_dim,)`` vector in the schema's column order."""
        if self.strict_schema:
            unexpected = set(features.keys()) - set(self.feature_names)
            if unexpected:
                raise ValueError(
                    f"feature dict has {len(unexpected)} keys not in the "
                    f"trained schema; first few: {sorted(unexpected)[:5]}. "
                    "Pass strict_schema=False to ignore."
                )

        n = len(self.feature_names)
        x = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(self.feature_names):
            v = features.get(name)
            if v is None:
                continue
            try:
                x[i] = float(v)
            except (TypeError, ValueError):
                x[i] = 0.0
        # Replace NaN/Inf to keep the model healthy.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # StandardScaler-style normalize.
        return ((x - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    def _predict_single(self, x: np.ndarray) -> Diagnosis:
        import torch

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0)  # (1, input_dim)
            z, h_groups = self.model.encode(x_t)

            det_logits = self.model.detect(z)
            det_probs = torch.softmax(det_logits, dim=-1)[0]
            faulty_prob = float(det_probs[1].item())
            is_faulty = bool(det_logits.argmax(dim=-1).item() == 1)

            cat_logits = self.model.categorize(z)
            cat_probs = torch.softmax(cat_logits, dim=-1)[0]
            cat_idx = int(cat_logits.argmax(dim=-1).item())
            cat_name = (self.category_names[cat_idx]
                        if 0 <= cat_idx < len(self.category_names) else None)
            cat_prob = float(cat_probs[cat_idx].item()) if cat_name else 0.0

            rc_name: Optional[str] = None
            rc_prob = 0.0
            group_importance: dict[str, float] = {}
            if is_faulty and cat_name is not None:
                rc_name, rc_prob, group_importance = self._stage3(
                    z, h_groups, cat_name,
                )

        return Diagnosis(
            is_faulty=is_faulty,
            detection_prob=faulty_prob,
            category=cat_name if is_faulty else None,
            category_prob=cat_prob if is_faulty else 0.0,
            root_cause=rc_name,
            root_cause_prob=rc_prob,
            group_importance=group_importance,
        )

    def _stage3(self, z, h_groups, cat_name: str
                ) -> tuple[Optional[str], float, dict[str, float]]:
        """Return (root_cause, prob, group_importance) for one sample."""
        import torch

        rc_logits = self.model.diagnose(z, cat_name)
        if rc_logits is None:
            return None, 0.0, {}
        rc_probs = torch.softmax(rc_logits, dim=-1)[0]
        rc_idx = int(rc_logits.argmax(dim=-1).item())
        names = self.rootcause_names.get(cat_name, [])
        rc_name = names[rc_idx] if 0 <= rc_idx < len(names) else f"rc_{rc_idx}"
        rc_prob = float(rc_probs[rc_idx].item())

        # Optional group importance from the prototype matcher.
        group_importance: dict[str, float] = {}
        try:
            preds, _, group_dists = self.model.diagnose_proto(h_groups, cat_name)
        except Exception:
            preds, group_dists = None, None
        if preds is not None and group_dists is not None:
            # group_dists: (1, n_rc, n_groups). Compare predicted vs
            # nearest alternative within the same category.
            gd = group_dists[0]  # (n_rc, n_groups)
            pred_idx = int(preds[0].item())
            if gd.shape[0] >= 2:
                alt_total = gd.sum(dim=-1).clone()
                alt_total[pred_idx] = float("inf")
                alt_idx = int(alt_total.argmin().item())
                margin = (gd[alt_idx] - gd[pred_idx]).cpu().numpy()
                for i, name in enumerate(self.group_names):
                    if i < len(margin):
                        group_importance[name] = float(margin[i])
        return rc_name, rc_prob, group_importance


# ─────────────────────────────────────────────────────────────────────────
# Model builder — lives outside the class so the import path stays
# isolated. The training-side ``HierarchicalDiagnosisModel`` is reachable
# from this layout because ``hierarchical_graph_category_rootcause`` sits
# at the package root in the source tree.
# ─────────────────────────────────────────────────────────────────────────
def _build_model_from_kwargs(model_kwargs: Mapping[str, Any]):
    """Instantiate a ``HierarchicalDiagnosisModel`` from saved kwargs."""
    try:
        # The research-side training driver lives outside the wheel;
        # we import it lazily so the package itself doesn't blow up if
        # the research side hasn't been pip-installed.
        from hierarchical_graph_category_rootcause.model import (
            HierarchicalDiagnosisModel,
        )
    except ImportError as exc:  # pragma: no cover - install issue
        raise ImportError(
            "defaultplusplus.diagnosis requires the "
            "``hierarchical_graph_category_rootcause`` module to be on "
            "PYTHONPATH. From a source checkout this is the package root; "
            "install the research stack via 'pip install -e .[all]'."
        ) from exc

    return HierarchicalDiagnosisModel(**dict(model_kwargs))
