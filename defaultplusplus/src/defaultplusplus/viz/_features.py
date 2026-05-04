"""Small parsers for the feature-key conventions emitted by
``FeatureExtractor.finalize()`` and the offline raw collector.

Two layer conventions appear in the wild:

  short form:  ``ffn_delta_l3_phase_early_mean``
  long form:   ``grad_norm_layer3_attention__epoch_mean__phase_early_mean``

Both forms are routed correctly by ``feature_groups.assign_feature_to_group``
(updated to accept either). The viz layer needs the same flexibility:
parsing a per-layer family means knowing which characters in the key
encode the layer index and which encode the metric base name.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

# Group 1: metric prefix; group 2: layer index; group 3: rest
_LAYER_RE = re.compile(r"^(.*?)_l(?:ayer)?(\d+)_(.*)$")

# Phase suffix recognized in offline raw CSVs
_PHASE_RE = re.compile(
    r"__epoch_(mean|std)__phase_(early|mid|final|slope)(?:_mean|_value)?$"
)


def parse_layer_key(key: str) -> tuple[str, int, str] | None:
    """Return ``(metric_base, layer_idx, suffix)`` if ``key`` is per-layer.

    Returns ``None`` when the key has no embedded layer index.
    """
    m = _LAYER_RE.match(key)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def per_layer_families(keys: Iterable[str]) -> dict[str, dict[int, str]]:
    """Group keys by ``(metric_base, suffix)`` and index by layer.

    Returns ``{family_name: {layer_idx: full_key}}`` so the per-layer
    heatmap can read one row per family without scanning the full key
    list multiple times.
    """
    families: dict[str, dict[int, str]] = defaultdict(dict)
    for k in keys:
        parsed = parse_layer_key(k)
        if parsed is None:
            continue
        base, layer, suffix = parsed
        family = f"{base}__{suffix}" if suffix else base
        families[family][layer] = k
    # Drop families with only one layer (nothing to heatmap)
    return {f: ls for f, ls in families.items() if len(ls) > 1}


def split_phase(key: str) -> tuple[str, str | None]:
    """Strip the ``__epoch_*__phase_*`` suffix.

    Returns ``(stem, phase_label)`` where ``phase_label`` is something
    like ``"early_mean"`` / ``"final_value"`` / ``None`` if the key has
    no recognizable phase suffix.
    """
    m = _PHASE_RE.search(key)
    if not m:
        return key, None
    stem = key[: m.start()]
    return stem, m.group(2)


def find_keys_matching(features: dict, *needles: str) -> list[str]:
    """Return feature keys containing every needle (case-insensitive)."""
    out: list[str] = []
    for k in features:
        kl = k.lower()
        if all(n.lower() in kl for n in needles):
            out.append(k)
    return sorted(out)
