"""Tests for the cache_nll_divergence probe (T14, decoder-only)."""
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
    return model, tok


def _encode(tok, max_length: int = 16) -> dict:
    text = ["the quick brown fox jumps over the lazy dog"] * 2
    enc = tok(text, padding=True, truncation=True, return_tensors="pt",
              max_length=max_length)
    return enc


def test_cache_nll_divergence_is_near_zero_on_clean_decoder() -> None:
    model, tok = _build_tiny_gpt2()
    enc = _encode(tok)

    from defaultplusplus.config import ExtractionConfig
    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.metrics.cache import CacheMetrics

    inspector = ModelInspector(model)
    metrics = CacheMetrics(inspector, ExtractionConfig())

    div = metrics._compute_cache_nll_divergence(
        model=model,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    assert div is not None, "probe should run on a tiny GPT-2"
    # Float64 KL on identical distributions is bounded by ~1e-12 in
    # practice; we use a generous bound to absorb numerical noise.
    assert div < 1e-6, f"clean cache divergence too large: {div!r}"


def test_cache_nll_divergence_responds_to_corrupted_cached_path() -> None:
    """If the cached forward is perturbed, the symmetric KL must rise."""
    model, tok = _build_tiny_gpt2()
    enc = _encode(tok)

    from defaultplusplus.config import ExtractionConfig
    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.metrics.cache import CacheMetrics

    inspector = ModelInspector(model)
    metrics = CacheMetrics(inspector, ExtractionConfig())

    clean = metrics._compute_cache_nll_divergence(
        model=model,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    assert clean is not None and clean < 1e-6

    torch.manual_seed(0)
    original_forward = model.forward

    def _faulty_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if kwargs.get("past_key_values") is not None and hasattr(out, "logits"):
            out.logits = out.logits + torch.randn_like(out.logits) * 5.0
        return out

    model.forward = _faulty_forward
    try:
        faulty = metrics._compute_cache_nll_divergence(
            model=model,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
    finally:
        model.forward = original_forward

    assert faulty is not None
    assert faulty > 1.0, f"perturbed cache should diverge measurably; got {faulty!r}"
    assert faulty - clean > 1.0


def test_cache_probe_respects_interval() -> None:
    """Probe runs only at multiples of ``cache_probe_interval``."""
    model, tok = _build_tiny_gpt2()
    enc = _encode(tok)

    from defaultplusplus.config import ExtractionConfig
    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.metrics.cache import CacheMetrics

    inspector = ModelInspector(model)
    cfg = ExtractionConfig(cache_probe_interval=10)
    metrics = CacheMetrics(inspector, cfg)

    # batch_idx=0 is divisible by 10 -> probe runs (or returns 0 fall-back).
    out_zero = metrics.collect(
        model=model, outputs=None,
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        batch_idx=0,
    )
    assert "cache_nll_divergence" in out_zero

    # batch_idx=3 is not on the cadence -> probe is gated and stays 0.
    out_skip = metrics.collect(
        model=model, outputs=None,
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        batch_idx=3,
    )
    assert out_skip["cache_nll_divergence"] == 0.0


def test_collector_emits_cache_nll_divergence_for_decoder() -> None:
    """End-to-end: FeatureExtractor on a tiny decoder emits the key."""
    model, tok = _build_tiny_gpt2()
    enc = _encode(tok)
    enc["labels"] = enc["input_ids"].clone()

    from defaultplusplus import FeatureExtractor

    with FeatureExtractor(model, arch="decoder") as fx:
        outputs = model(**enc)
        loss = outputs.loss
        loss.backward()
        metrics = fx.collector.collect_step(
            loss=loss, model=model, optimizer=None, outputs=outputs,
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            labels=enc["labels"], batch_idx=0, epoch=0, step_time=0.01,
        )
    assert "cache_nll_divergence" in metrics
    assert metrics["cache_nll_divergence"] >= 0.0
