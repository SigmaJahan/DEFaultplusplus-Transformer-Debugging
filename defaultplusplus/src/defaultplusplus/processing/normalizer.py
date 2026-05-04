"""Single-run anomaly encoding.

The offline benchmark pipeline trains the diagnostic model on paired
clean / faulty traces, but the runtime API is a single live run. We
close that gap with a learned **clean reference**: per-key
``(median, mad, std, count)`` summaries computed from the baseline
(``is_faulty == 0``) subset of the training data.

At runtime, :class:`RuntimeNormalizer.encode` takes the dict returned by
``FeatureExtractor.finalize()`` and produces a dict in the trained
diagnoser's schema:

  * Every key the model expects is present (missing keys are filled
    with the reference median, so the diagnoser never sees a zero
    where a real-world median would carry signal).
  * Optional: ``mode="anomaly"`` returns ``(value − median) / mad``
    z-scores instead of raw absolute values; useful when the consumer
    wants to highlight deviation explicitly. The default ``mode="raw"``
    is what the shipped diagnostic checkpoint expects, since training
    used absolute values, not deltas.
  * Aliases map runtime short-form layer names (``..._l3_...``) to
    long-form trained-schema names (``..._layer3_...``) and vice versa.
    Either convention round-trips.

The fitted reference is a small numpy archive (~50 KB per arch) and
ships next to the pretrained weights under
``defaultplusplus/pretrained/weights/{arch}_reference.npz``.

To re-fit (offline, after collecting more baseline runs)::

    python scripts/fit_runtime_reference.py \\
        --arch encoder \\
        --csv data/paper-aligned-csv/encoder_merged.csv \\
        --output src/defaultplusplus/pretrained/weights/encoder_reference.npz
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

# MAD scale factor for normal distribution: std ≈ 1.4826 * MAD.
_MAD_TO_SIGMA = 1.4826

# Short-form (in-process extractor) and long-form (offline raw collector)
# layer prefixes are the same metric. Round-trip both.
_SHORT_LAYER_RE = re.compile(r"^(.*?)_l(\d+)_(.*)$")
_LONG_LAYER_RE = re.compile(r"^(.*?)_layer(\d+)_(.*)$")
_TRACE_PREFIX = "trace__"


@dataclass(frozen=True)
class RuntimeReference:
    """The clean-run summary that anchors anomaly encoding.

    Stored as a flat ``.npz`` so it can ship in the wheel without
    pickle. Fields:

      schema:     ordered list of feature names the diagnoser expects.
      median:     per-key baseline median.
      mad:        per-key median absolute deviation (robust scale).
      std:        per-key baseline standard deviation (auxiliary).
      n_baseline: number of baseline rows that built the reference.
      arch:       ``"encoder"`` or ``"decoder"``.
    """
    schema: list[str]
    median: np.ndarray  # (n,)
    mad: np.ndarray     # (n,)
    std: np.ndarray     # (n,)
    n_baseline: int
    arch: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            schema=np.asarray(self.schema, dtype=object),
            median=self.median.astype(np.float32),
            mad=self.mad.astype(np.float32),
            std=self.std.astype(np.float32),
            n_baseline=np.int64(self.n_baseline),
            arch=np.asarray(self.arch, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeReference":
        z = np.load(Path(path), allow_pickle=True)
        return cls(
            schema=[str(s) for s in z["schema"].tolist()],
            median=np.asarray(z["median"], dtype=np.float64),
            mad=np.asarray(z["mad"], dtype=np.float64),
            std=np.asarray(z["std"], dtype=np.float64),
            n_baseline=int(z["n_baseline"]),
            arch=str(z["arch"]),
        )


def fit_reference(X: np.ndarray, feature_names: list[str], arch: str,
                  ) -> RuntimeReference:
    """Fit a reference from a baseline-only feature matrix.

    Args:
        X:              ``(n_baseline, n_features)`` float matrix.
        feature_names:  column names matching ``X.shape[1]``.
        arch:           ``"encoder"`` or ``"decoder"``.

    NaN cells are ignored per-column via ``np.nanmedian`` /
    ``np.nanstd`` so columns with sporadic missingness still contribute.
    """
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X.shape[1]} cols but {len(feature_names)} names"
        )
    if X.shape[0] == 0:
        raise ValueError("need at least one baseline row to fit a reference")

    median = np.nanmedian(X, axis=0)
    deviations = np.abs(X - median[None, :])
    mad = np.nanmedian(deviations, axis=0) * _MAD_TO_SIGMA
    std = np.nanstd(X, axis=0)

    # Floor MAD so we never divide by zero in encode().
    mad = np.where(mad > 1e-9, mad, np.where(std > 1e-9, std, 1.0))
    std = np.where(std > 1e-9, std, 1.0)

    median = np.nan_to_num(median, nan=0.0)
    mad = np.nan_to_num(mad, nan=1.0)
    std = np.nan_to_num(std, nan=1.0)

    return RuntimeReference(
        schema=list(feature_names),
        median=median.astype(np.float64),
        mad=mad.astype(np.float64),
        std=std.astype(np.float64),
        n_baseline=int(X.shape[0]),
        arch=arch,
    )


def _alias_keys(features: Mapping[str, float]) -> dict[str, float]:
    """Expand each runtime key to both short and long layer-form
    variants so a downstream lookup against either convention succeeds.

    Also strips the optional ``trace__`` prefix on each key.
    """
    out: dict[str, float] = {}
    for k, v in features.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        # Drop the trace__ prefix (a runtime-only namespace) so trace
        # keys can match against the offline schema.
        bare = k[len(_TRACE_PREFIX):] if k.startswith(_TRACE_PREFIX) else k
        out.setdefault(bare, fv)
        out.setdefault(k, fv)

        m = _SHORT_LAYER_RE.match(bare)
        if m:
            base, layer, rest = m.group(1), m.group(2), m.group(3)
            out.setdefault(f"{base}_layer{layer}_{rest}", fv)
            continue
        m = _LONG_LAYER_RE.match(bare)
        if m:
            base, layer, rest = m.group(1), m.group(2), m.group(3)
            out.setdefault(f"{base}_l{layer}_{rest}", fv)
    return out


class RuntimeNormalizer:
    """Convert a single live run's feature dict into model-ready shape.

    Construct via :meth:`load` (uses the reference shipped with
    :func:`defaultplusplus.diagnosis.weights_path`) or build a
    :class:`RuntimeReference` yourself and pass it to ``__init__``.

    The class is stateless beyond the bundled reference; multiple
    threads can share one instance.
    """

    def __init__(self, reference: RuntimeReference) -> None:
        self.reference = reference
        self._index = {k: i for i, k in enumerate(reference.schema)}

    @classmethod
    def load(cls, arch: str, *,
             reference: Optional[Path | str] = None) -> "RuntimeNormalizer":
        """Load the reference shipped with the package.

        ``arch`` is ``"encoder"`` or ``"decoder"``. When ``reference``
        is omitted we look under
        ``pretrained/weights/{arch}_reference.npz`` next to the
        diagnostic-model checkpoints.
        """
        path = Path(reference) if reference is not None else _default_path(arch)
        if not path.exists():
            raise FileNotFoundError(
                f"no runtime reference at {path}. Fit one with "
                f"scripts/fit_runtime_reference.py and place it next to "
                f"the diagnostic-model checkpoints."
            )
        return cls(RuntimeReference.load(path))

    def encode(self,
               features: Mapping[str, float],
               *,
               mode: str = "raw") -> dict[str, float]:
        """Return a feature dict keyed exactly by the diagnoser schema.

        Args:
            features: live feature dict, typically the output of
                ``FeatureExtractor.finalize()``. Both short-form
                (``..._l3_...``) and long-form (``..._layer3_...``)
                layer keys are accepted.
            mode: ``"raw"`` (default) returns the input value clamped
                onto the schema, with missing keys filled with the
                reference median. ``"anomaly"`` returns
                ``(value − median) / mad`` z-scores; missing keys
                become 0.0 (definitionally "no deviation").

        Returns:
            ``dict[str, float]`` with exactly ``reference.schema`` keys.
        """
        if mode not in ("raw", "anomaly"):
            raise ValueError(
                f"mode must be 'raw' or 'anomaly', got {mode!r}"
            )
        aliased = _alias_keys(features)
        out: dict[str, float] = {}
        ref = self.reference
        n_present = 0
        for i, key in enumerate(ref.schema):
            v = aliased.get(key)
            if v is None:
                if mode == "raw":
                    out[key] = float(ref.median[i])
                else:
                    out[key] = 0.0
            else:
                n_present += 1
                if mode == "raw":
                    out[key] = float(v)
                else:
                    out[key] = float(
                        (v - ref.median[i]) / ref.mad[i]
                    )
        # Cache last-coverage for diagnostics; harmless mutable.
        self._last_coverage = (n_present, len(ref.schema))
        return out

    def coverage(self,
                 features: Mapping[str, float]) -> tuple[int, int]:
        """How many schema keys ``features`` would populate (raw, not aliased).

        Returns ``(n_matched, n_total)``. Useful as a smoke check: if
        coverage is suspiciously low, the runtime extractor and the
        diagnoser were probably trained against different schema
        versions.
        """
        aliased = _alias_keys(features)
        n_match = sum(1 for k in self.reference.schema if k in aliased)
        return n_match, len(self.reference.schema)


def _default_path(arch: str) -> Path:
    if arch not in ("encoder", "decoder"):
        raise ValueError(f"unknown arch {arch!r}")
    here = Path(__file__).resolve().parent.parent
    return here / "pretrained" / "weights" / f"{arch}_reference.npz"
