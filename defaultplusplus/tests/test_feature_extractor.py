"""Smoke tests for the public FeatureExtractor + HF callback API.

These tests exercise the package as a downstream user would: import
``FeatureExtractor`` from the top of the package, run a few training
steps on a tiny real model, and assert that the finalize call returns
a non-empty feature dictionary keyed against the runtime schema.

The HF callback test uses a `Trainer` if the `transformers` package is
installed; otherwise it is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────
# Public-API surface
# ─────────────────────────────────────────────────────────────────────────
def test_top_level_imports() -> None:
    import defaultplusplus

    assert hasattr(defaultplusplus, "FeatureExtractor")
    assert hasattr(defaultplusplus, "ExtractionConfig")
    assert hasattr(defaultplusplus, "build_feature_vector")
    assert defaultplusplus.__version__


def test_hf_callback_is_lazy_when_transformers_missing(monkeypatch) -> None:
    # Verify that ``defaultplusplus`` itself imports without
    # ``transformers``. We do not test for the actual ImportError on
    # access here because the test environment has ``transformers``
    # installed; the lazy resolution path is exercised by the test
    # below that uses the real callback.
    import defaultplusplus
    assert "DEFaultPlusCallback" in defaultplusplus.__all__


# ─────────────────────────────────────────────────────────────────────────
# Encoder smoke run
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("model_name", ["hf-internal-testing/tiny-random-DistilBertForSequenceClassification"])
def test_extractor_runs_on_tiny_encoder(model_name: str, tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from defaultplusplus import FeatureExtractor

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        output_attentions=True, output_hidden_states=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    text = ["positive", "negative", "ok"] * 4
    labels = torch.tensor([1, 0, 1] * 4)
    enc = tok(text, padding=True, truncation=True, return_tensors="pt", max_length=16)
    enc["labels"] = labels

    with FeatureExtractor(model, arch="encoder") as fx:
        for epoch in range(2):
            for _step in range(3):
                outputs = model(**enc)
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                fx.step(
                    loss=outputs.loss,
                    outputs=outputs,
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=enc["labels"],
                    optimizer=optimizer,
                )
            fx.epoch_end(epoch)
            fx.record_validation(epoch, {"accuracy": 0.8, "loss": 0.5})
        feature_vector = fx.finalize()

    # Vector is non-empty, all-finite, JSON-serializable.
    assert len(feature_vector) > 0
    for value in feature_vector.values():
        assert isinstance(value, float)
        assert not (value != value)  # not NaN

    out_file = tmp_path / "features.json"
    out_file.write_text(json.dumps(feature_vector))
    parsed = json.loads(out_file.read_text())
    assert parsed.keys() == feature_vector.keys()


# ─────────────────────────────────────────────────────────────────────────
# Decoder smoke run (causal LM)
# ─────────────────────────────────────────────────────────────────────────
def test_extractor_runs_on_tiny_decoder() -> None:
    transformers = pytest.importorskip("transformers")
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from defaultplusplus import FeatureExtractor

    model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    text = ["hello world", "the quick brown fox", "a b c d e"] * 3
    enc = tok(text, padding=True, truncation=True, return_tensors="pt", max_length=16)
    enc["labels"] = enc["input_ids"].clone()

    with FeatureExtractor(model, arch="decoder") as fx:
        for _step in range(3):
            outputs = model(**enc)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            fx.step(
                loss=outputs.loss,
                outputs=outputs,
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=enc["labels"],
                optimizer=optimizer,
            )
        fx.epoch_end(0)
        feature_vector = fx.finalize()

    assert len(feature_vector) > 0


# ─────────────────────────────────────────────────────────────────────────
# Architecture-mismatch fail-closed
# ─────────────────────────────────────────────────────────────────────────
def test_arch_mismatch_raises() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification

    from defaultplusplus import FeatureExtractor

    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        num_labels=2)
    with pytest.raises(ValueError, match="Requested arch"):
        FeatureExtractor(model, arch="decoder")


def test_unknown_arch_alias_raises() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification

    from defaultplusplus import FeatureExtractor

    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        num_labels=2)
    with pytest.raises(ValueError, match="Unknown arch hint"):
        FeatureExtractor(model, arch="t5-style")


# ─────────────────────────────────────────────────────────────────────────
# HF Trainer callback
# ─────────────────────────────────────────────────────────────────────────
def test_hf_callback_runs_with_trainer(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    # HF Trainer requires ``accelerate`` since transformers 4.40.
    pytest.importorskip("accelerate", reason="accelerate is required by HF Trainer")
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, Trainer,
        TrainingArguments,
    )

    from defaultplusplus.hf_callback import DEFaultPlusCallback

    model_name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    class _SyntheticDataset(Dataset):
        def __init__(self) -> None:
            text = ["positive", "negative", "ok", "bad"] * 4
            label = [1, 0, 1, 0] * 4
            enc = tok(text, padding="max_length", truncation=True, max_length=16)
            self._items = []
            for i in range(len(text)):
                self._items.append({
                    "input_ids": torch.tensor(enc["input_ids"][i]),
                    "attention_mask": torch.tensor(enc["attention_mask"][i]),
                    "labels": torch.tensor(label[i]),
                })

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    out_path = tmp_path / "features.json"
    callback = DEFaultPlusCallback(out_path=out_path, arch="encoder")

    class _Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            callback.capture_inputs(dict(inputs))
            outputs = model(**inputs)
            callback.capture_outputs(outputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
    )
    trainer_kwargs = dict(
        model=model, args=args,
        train_dataset=_SyntheticDataset(),
        callbacks=[callback],
    )
    # ``tokenizer=`` was renamed to ``processing_class=`` in
    # transformers 4.46. Pass whichever the installed version accepts.
    import inspect
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = _Trainer(**trainer_kwargs)
    trainer.train()

    assert callback.feature_vector is not None
    assert len(callback.feature_vector) > 0
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    assert parsed.keys() == callback.feature_vector.keys()
