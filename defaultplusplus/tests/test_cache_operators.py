"""Tests for the four KV-cache operator implementations: CTR, COB, CST, CLK.

Each operator should:
  * mutate the cache visibly during a cached forward pass,
  * leave the model parameters untouched (no static side-effect),
  * restore the original ``forward`` after exit.

We verify both via the ``cache_nll_divergence`` probe (end-to-end signal)
and via direct inspection of the cache tensors.
"""
from __future__ import annotations

import pytest
import torch


def _build_tiny_gpt2():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model, tok


def _encoded(tok, max_length: int = 16):
    text = ["the quick brown fox jumps over the lazy dog"] * 2
    return tok(text, padding=True, truncation=True, return_tensors="pt",
               max_length=max_length)


def _clean_baseline(model, enc) -> float:
    from defaultplusplus.config import ExtractionConfig
    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.metrics.cache import CacheMetrics

    inspector = ModelInspector(model)
    metrics = CacheMetrics(inspector, ExtractionConfig())
    return metrics._compute_cache_nll_divergence(
        model=model, input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )


def _faulty_divergence(model, enc, op_id: str, **kwargs) -> float:
    from defaultplusplus.config import ExtractionConfig
    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.metrics.cache import CacheMetrics
    from defaultplusplus.deform import get_injector

    inspector = ModelInspector(model)
    metrics = CacheMetrics(inspector, ExtractionConfig())
    cls = get_injector(op_id, **kwargs)
    with cls(model):
        return metrics._compute_cache_nll_divergence(
            model=model, input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )


# ─────────────────────────────────────────────────────────────────────────
# CTR — truncate cache
# ─────────────────────────────────────────────────────────────────────────
def test_ctr_moves_cache_nll_divergence() -> None:
    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)

    clean = _clean_baseline(model, enc)
    faulty = _faulty_divergence(model, enc, "CTR", param_value=4)

    assert clean < 1e-9
    assert faulty > 100 * max(clean, 1e-12), \
        f"CTR should move divergence; got clean={clean!r} faulty={faulty!r}"


def test_ctr_truncates_cache_in_place() -> None:
    """Direct check: after CTR runs, the cache shrinks to ``length`` rows."""
    from defaultplusplus.deform.operator_impls.registry import (
        _iter_cache_layers, _truncate_cache,
    )

    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)
    with torch.no_grad():
        warm = model(input_ids=enc["input_ids"][:, :8], use_cache=True)
    cache = warm.past_key_values

    _truncate_cache(cache, 3)
    seqs = [k.shape[-2] for _, k, _ in _iter_cache_layers(cache)]
    assert seqs and all(s == 3 for s in seqs), seqs


# ─────────────────────────────────────────────────────────────────────────
# COB — off-by-one cache index
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("shift", [-1, 1])
def test_cob_moves_cache_nll_divergence(shift: int) -> None:
    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)

    clean = _clean_baseline(model, enc)
    faulty = _faulty_divergence(model, enc, "COB", param_value=shift)

    assert clean < 1e-9
    assert faulty > 100 * max(clean, 1e-12), \
        f"COB(shift={shift}) should move divergence; clean={clean!r} faulty={faulty!r}"


def test_cob_zeros_vacated_position() -> None:
    """COB shift=-1 must zero out the last cache position after the shift."""
    from defaultplusplus.deform.operator_impls.registry import (
        _iter_cache_layers, _shift_cache,
    )

    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)
    with torch.no_grad():
        warm = model(input_ids=enc["input_ids"][:, :6], use_cache=True)
    cache = warm.past_key_values

    _shift_cache(cache, -1)
    for _, k, v in _iter_cache_layers(cache):
        # After shift=-1 the trailing row is zero-padded.
        assert torch.allclose(k[..., -1, :], torch.zeros_like(k[..., -1, :]))
        assert torch.allclose(v[..., -1, :], torch.zeros_like(v[..., -1, :]))


# ─────────────────────────────────────────────────────────────────────────
# CST — stale (one-step-old) cache
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("scope", ["all", "first", "last", "middle"])
def test_cst_moves_cache_nll_divergence(scope: str) -> None:
    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)

    clean = _clean_baseline(model, enc)
    faulty = _faulty_divergence(model, enc, "CST", param_value=scope)

    assert clean < 1e-9
    assert faulty > 100 * max(clean, 1e-12), \
        f"CST(scope={scope}) should move divergence; clean={clean!r} faulty={faulty!r}"


def test_cst_drops_one_position_per_targeted_layer() -> None:
    """CST scope='first' must shorten only layer 0's cache."""
    from defaultplusplus.deform.operator_impls.registry import (
        _iter_cache_layers, _stale_cache,
    )

    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)
    with torch.no_grad():
        warm = model(input_ids=enc["input_ids"][:, :8], use_cache=True)
    cache = warm.past_key_values

    before = [k.shape[-2] for _, k, _ in _iter_cache_layers(cache)]
    _stale_cache(cache, "first")
    after = [k.shape[-2] for _, k, _ in _iter_cache_layers(cache)]
    assert after[0] == before[0] - 1, (before, after)
    for i in range(1, len(after)):
        assert after[i] == before[i], (i, before, after)


# ─────────────────────────────────────────────────────────────────────────
# CLK — cross-request cache leak
# ─────────────────────────────────────────────────────────────────────────
def test_clk_moves_cache_nll_divergence() -> None:
    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)

    clean = _clean_baseline(model, enc)
    faulty = _faulty_divergence(model, enc, "CLK")

    assert clean < 1e-9
    # CLK is the most disruptive of the four — divergence should clear
    # the noise floor by many orders of magnitude.
    assert faulty > 1e-4, f"CLK should leak measurably; got {faulty!r}"


def test_clk_stashes_and_replays_cache_reference() -> None:
    """After a forward, CLK should hold a cache stash; a subsequent
    cache-less call must receive the stashed cache as past_key_values."""
    from defaultplusplus.deform import get_injector

    model, tok = _build_tiny_gpt2()
    enc = _encoded(tok)
    inj = get_injector("CLK")(model)

    captured_kwargs = []
    with inj:
        # Drive one forward to populate the stash.
        with torch.no_grad():
            model(input_ids=enc["input_ids"][:, :4], use_cache=True)
        assert inj._cache_stash is not None, "stash should be populated"

        # Patch the wrapper to spy on what it forwards downstream.
        original_mutate_cache = type(inj)._mutate_cache_inputs

        def _spy(self, kwargs):
            out = original_mutate_cache(self, kwargs)
            captured_kwargs.append(dict(out))
            return out
        type(inj)._mutate_cache_inputs = _spy
        try:
            with torch.no_grad():
                model(input_ids=enc["input_ids"][:, :4], use_cache=True)
        finally:
            type(inj)._mutate_cache_inputs = original_mutate_cache

    assert captured_kwargs, "wrapper should have been invoked"
    leaked = captured_kwargs[-1]
    assert "past_key_values" in leaked
    assert leaked["past_key_values"] is not None


# ─────────────────────────────────────────────────────────────────────────
# Restoration: every cache op must leave model.forward unchanged on exit.
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("op_id, kwargs", [
    ("CTR", {"param_value": 4}),
    ("COB", {"param_value": 1}),
    ("CST", {"param_value": "all"}),
    ("CLK", {}),
])
def test_cache_operator_restores_forward(op_id: str, kwargs: dict) -> None:
    """The injector must rebind ``model.forward`` on enter and restore it on exit.

    Bound methods are recreated on every attribute access, so we compare
    via :func:`_callable_identity` — the same helper the structural
    verifier uses.
    """
    from defaultplusplus.deform import get_injector
    from defaultplusplus.deform.validation import _callable_identity

    model, _ = _build_tiny_gpt2()
    before = _callable_identity(model.forward)
    with get_injector(op_id, **kwargs)(model):
        assert _callable_identity(model.forward) != before, \
            f"{op_id} did not rebind model.forward inside the context"
    assert _callable_identity(model.forward) == before, \
        f"{op_id} did not restore model.forward on exit"
