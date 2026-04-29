"""Tests for the sublayer-boundary hook plumbing (T13 + T15)."""
from __future__ import annotations

import pytest
import torch


def _build_tiny_distilbert():
    """Return a tiny distilbert encoder + tokenizer; skip if HF missing."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=2,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.eval()
    return model, tok


# ─────────────────────────────────────────────────────────────────────────
# T15: SublayerCapture installation, capture, and removal lifecycle
# ─────────────────────────────────────────────────────────────────────────
def test_sublayer_capture_records_ffn_and_ln_taps() -> None:
    model, tok = _build_tiny_distilbert()
    enc = tok("hello world", padding=True, return_tensors="pt", max_length=8, truncation=True)

    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.sublayer_capture import SublayerCapture

    inspector = ModelInspector(model)
    capture = SublayerCapture(inspector)
    capture.install()
    try:
        with torch.no_grad():
            model(**enc)
        # Every layer should have ffn_in / ffn_out and at least one ln out.
        for layer_idx in range(inspector.num_layers):
            assert capture.has(layer_idx, "ffn_in"), f"layer {layer_idx} missing ffn_in"
            assert capture.has(layer_idx, "ffn_out"), f"layer {layer_idx} missing ffn_out"
            ffn_in = capture.get(layer_idx, "ffn_in")
            ffn_out = capture.get(layer_idx, "ffn_out")
            assert ffn_in.shape[-1] == inspector.hidden_size
            assert ffn_out.shape[-1] == inspector.hidden_size
            assert any(capture.has(layer_idx, f"ln{i}_out") for i in range(4))
    finally:
        capture.remove()
    assert not capture.installed
    assert capture.captures == {}


def test_sublayer_capture_records_qkv_taps() -> None:
    model, tok = _build_tiny_distilbert()
    enc = tok("hi", padding=True, return_tensors="pt", max_length=8, truncation=True)

    from defaultplusplus.extraction.inspector import ModelInspector
    from defaultplusplus.extraction.sublayer_capture import SublayerCapture

    inspector = ModelInspector(model)
    with SublayerCapture(inspector) as capture:
        with torch.no_grad():
            model(**enc)
        for layer_idx in range(inspector.num_layers):
            assert capture.has(layer_idx, "q"), f"layer {layer_idx} missing q"
            assert capture.has(layer_idx, "k"), f"layer {layer_idx} missing k"
            assert capture.has(layer_idx, "v"), f"layer {layer_idx} missing v"
            q = capture.get(layer_idx, "q")
            k = capture.get(layer_idx, "k")
            v = capture.get(layer_idx, "v")
            assert q.shape == k.shape == v.shape


# ─────────────────────────────────────────────────────────────────────────
# T15: end-to-end FeatureExtractor exposes exact LN / FFN metrics
# ─────────────────────────────────────────────────────────────────────────
def test_extractor_emits_exact_ln_and_ffn_metrics() -> None:
    model, tok = _build_tiny_distilbert()
    enc = tok(["alpha beta", "gamma delta"], padding=True, return_tensors="pt",
              max_length=8, truncation=True)

    from defaultplusplus import FeatureExtractor

    with FeatureExtractor(model, arch="encoder") as fx:
        outputs = model(**enc)
        loss = outputs.logits.sum()
        loss.backward()
        metrics = fx.collector.collect_step(
            loss=loss, model=model, optimizer=None, outputs=outputs,
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            labels=None, batch_idx=0, epoch=0, step_time=0.01,
        )

    # The exact-tap path should have produced a non-zero LN std reading
    # for each layer; reconstruction would also produce a value, but the
    # exact path must at least emit the keys.
    assert any(k.startswith("ffn_delta_l") and k.endswith("_mean") for k in metrics)
    assert any(k.startswith("ln_std_l") and k.endswith("_mean") for k in metrics)
    assert any(k.startswith("residual_cos_l") for k in metrics)


# ─────────────────────────────────────────────────────────────────────────
# T13: qkv_alignment_* keys appear in metrics when hooks are installed
# ─────────────────────────────────────────────────────────────────────────
def test_extractor_emits_qkv_alignment_metrics() -> None:
    model, tok = _build_tiny_distilbert()
    enc = tok(["alpha beta", "gamma delta"], padding=True, return_tensors="pt",
              max_length=8, truncation=True)

    from defaultplusplus import FeatureExtractor

    with FeatureExtractor(model, arch="encoder") as fx:
        outputs = model(**enc)
        loss = outputs.logits.sum()
        loss.backward()
        metrics = fx.collector.collect_step(
            loss=loss, model=model, optimizer=None, outputs=outputs,
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            labels=None, batch_idx=0, epoch=0, step_time=0.01,
        )

    for key in ("qkv_alignment_qk_cos_mean",
                "qkv_alignment_qv_cos_mean",
                "qkv_alignment_kv_cos_mean"):
        assert key in metrics, f"{key} missing from collector metrics"
        assert -1.0 <= metrics[key] <= 1.0


# ─────────────────────────────────────────────────────────────────────────
# T15: hooks are torn down on FeatureExtractor exit
# ─────────────────────────────────────────────────────────────────────────
def test_feature_extractor_removes_hooks_on_exit() -> None:
    model, tok = _build_tiny_distilbert()

    from defaultplusplus import FeatureExtractor

    with FeatureExtractor(model, arch="encoder") as fx:
        assert fx.collector.sublayer_capture.installed
    assert not fx.collector.sublayer_capture.installed
    assert fx.collector.sublayer_capture.captures == {}
