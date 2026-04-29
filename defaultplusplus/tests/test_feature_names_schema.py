"""Tests for the fixed ``feature_names`` schema.

The schema is the load-bearing piece for shipping a pretrained
classifier — it must:

  * be available before any ``collect_step`` is called,
  * stay stable across epoch counts (early/mid/late are *fractions*),
  * declare every val_* key any registered task can produce,
  * be exactly matched by the keys that ``finalize()`` returns
    (modulo ``trace__*`` Eq 7.19 features, which are an additive
    extension).
"""
from __future__ import annotations

from typing import Any

import pytest
import torch


def _build_tiny_distilbert():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2,
        output_attentions=True, output_hidden_states=True,
    )
    return model, tok


def _train_n_epochs(fx, model, optim, enc, n: int, val_dict: dict) -> None:
    for ep in range(n):
        out = model(**enc)
        out.loss.backward()
        optim.step(); optim.zero_grad()
        fx.step(
            loss=out.loss, outputs=out,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=enc["labels"],
            optimizer=optim, step_time=0.01,
        )
        fx.epoch_end(ep)
        fx.record_validation(ep, val_dict)


# ─────────────────────────────────────────────────────────────────────────
# Fractional windows
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("total,expected", [
    (0,  {"early": (1, 0), "mid": (1, 0), "late": (1, 0)}),
    (1,  {"early": (1, 1), "mid": (1, 1), "late": (1, 1)}),
    (2,  {"early": (1, 1), "mid": (1, 2), "late": (2, 2)}),
    (3,  {"early": (1, 1), "mid": (2, 2), "late": (3, 3)}),
    (10, {"early": (1, 3), "mid": (4, 6), "late": (7, 10)}),
    (50, {"early": (1, 16), "mid": (17, 32), "late": (33, 50)}),
])
def test_compute_window_ranges_splits_into_thirds(total, expected) -> None:
    from defaultplusplus.extraction.aggregator import compute_window_ranges
    assert compute_window_ranges(total) == expected


def test_window_features_use_fractional_thirds() -> None:
    """Same column names produced regardless of run length."""
    from defaultplusplus.extraction.aggregator import compute_window_features

    history = {
        "loss": [(epoch, 1.0 / epoch) for epoch in range(1, 11)],
    }
    f5 = compute_window_features(
        {"loss": [(e, 1.0 / e) for e in range(1, 6)]}, total_epochs=5,
    )
    f10 = compute_window_features(history, total_epochs=10)
    f50 = compute_window_features(
        {"loss": [(e, 1.0 / e) for e in range(1, 51)]}, total_epochs=50,
    )

    expected_keys = {
        "loss_early_mean", "loss_early_slope",
        "loss_mid_mean",   "loss_mid_slope",
        "loss_late_mean",  "loss_late_slope",
        "loss_final",
    }
    assert set(f5) == expected_keys, "5-epoch run produced unexpected keys"
    assert set(f10) == expected_keys, "10-epoch run produced unexpected keys"
    assert set(f50) == expected_keys, "50-epoch run produced unexpected keys"


# ─────────────────────────────────────────────────────────────────────────
# feature_names is available pre-step + stable across runs
# ─────────────────────────────────────────────────────────────────────────
def test_feature_names_available_before_any_collect_step() -> None:
    """A downstream classifier must be able to read the schema before
    ``step()`` has ever been called."""
    model, _ = _build_tiny_distilbert()

    from defaultplusplus import FeatureExtractor

    fx = FeatureExtractor(model, arch="encoder")
    names = fx.feature_names
    assert len(names) > 100, "fixed schema looks empty"
    # The list is sorted for stable serialization.
    assert names == sorted(names)


def test_feature_names_stable_across_epoch_counts() -> None:
    """Training for 3 vs 10 vs 50 epochs must produce the same schema.
    """
    model, tok = _build_tiny_distilbert()
    text = ["ok", "bad", "good"] * 4
    enc = tok(text, padding=True, truncation=True,
              return_tensors="pt", max_length=8)
    enc["labels"] = torch.tensor([1, 0, 1] * 4)

    from defaultplusplus import FeatureExtractor
    from torch.optim import AdamW

    schemas: list[set[str]] = []
    for n_epochs in (3, 10, 25):
        model.train()
        optim = AdamW(model.parameters(), lr=1e-4)
        with FeatureExtractor(model, arch="encoder") as fx:
            _train_n_epochs(fx, model, optim, enc, n_epochs,
                            {"accuracy": 0.85, "loss": 0.5})
            features = fx.finalize()
        schemas.append({k for k in features if not k.startswith("trace__")})

    assert schemas[0] == schemas[1] == schemas[2], (
        "feature_names schema drifted across epoch counts"
    )


def test_feature_names_matches_finalize_output_after_run() -> None:
    """The keys ``finalize()`` returns must equal the fixed list
    (plus the additive ``trace__*`` Eq 7.19 features)."""
    model, tok = _build_tiny_distilbert()
    text = ["ok", "bad", "good"] * 4
    enc = tok(text, padding=True, truncation=True,
              return_tensors="pt", max_length=8)
    enc["labels"] = torch.tensor([1, 0, 1] * 4)

    from defaultplusplus import FeatureExtractor
    from torch.optim import AdamW

    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    fx = FeatureExtractor(model, arch="encoder")
    expected = set(fx.feature_names)
    _train_n_epochs(fx, model, optim, enc, 5,
                    {"accuracy": 0.85, "loss": 0.5})
    features = fx.finalize()

    actual = {k for k in features if not k.startswith("trace__")}
    assert actual == expected, (
        f"finalize() schema mismatch with feature_names: "
        f"missing={sorted(expected - actual)[:5]} "
        f"unexpected={sorted(actual - expected)[:5]}"
    )


def test_feature_names_includes_every_task_val_metric() -> None:
    """The fixed schema must declare val_* keys for every task in
    the registry. Users running CoLA (MCC), STS-B (Pearson/Spearman),
    or any future task get a stable column set."""
    model, _ = _build_tiny_distilbert()

    from defaultplusplus import FeatureExtractor

    fx = FeatureExtractor(model, arch="encoder")
    names = set(fx.feature_names)

    expected_val_metrics = (
        "accuracy", "loss", "f1",  # SST-2 / MRPC / QQP
        "matthews_correlation",     # CoLA
        "pearson", "spearmanr",     # STS-B
    )
    for raw in expected_val_metrics:
        for win in ("early", "mid", "late"):
            assert f"val_{raw}_{win}_mean" in names, f"missing val_{raw}_{win}_mean"
            assert f"val_{raw}_{win}_slope" in names, f"missing val_{raw}_{win}_slope"
        assert f"val_{raw}_final" in names, f"missing val_{raw}_final"
        assert f"final_val_{raw}" in names, f"missing final_val_{raw}"


# ─────────────────────────────────────────────────────────────────────────
# finalize() pads schema columns the runtime didn't produce
# ─────────────────────────────────────────────────────────────────────────
def test_finalize_pads_unrecorded_val_keys_to_zero() -> None:
    """A user who only records ``accuracy`` (SST-2 case) must still
    get every val_* schema column in the output, padded to 0.0."""
    model, tok = _build_tiny_distilbert()
    text = ["ok", "bad"] * 4
    enc = tok(text, padding=True, truncation=True,
              return_tensors="pt", max_length=8)
    enc["labels"] = torch.tensor([1, 0] * 4)

    from defaultplusplus import FeatureExtractor
    from torch.optim import AdamW

    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    with FeatureExtractor(model, arch="encoder") as fx:
        expected = set(fx.feature_names)
        _train_n_epochs(fx, model, optim, enc, 5, {"accuracy": 0.85})
        features = fx.finalize()

    actual = {k for k in features if not k.startswith("trace__")}
    assert actual == expected, "padded output must equal the fixed schema"

    # The val_* metrics the user did NOT record must be present at 0.0.
    assert features["val_pearson_early_mean"] == 0.0
    assert features["val_matthews_correlation_final"] == 0.0
    assert features["final_val_f1"] == 0.0


# ─────────────────────────────────────────────────────────────────────────
# validate_feature_names raises with diff
# ─────────────────────────────────────────────────────────────────────────
def test_validate_feature_names_passes_on_matching_schema() -> None:
    model, _ = _build_tiny_distilbert()

    from defaultplusplus import FeatureExtractor

    fx = FeatureExtractor(model, arch="encoder")
    expected = list(fx.feature_names)
    fx.collector.validate_feature_names(expected)  # must not raise


def test_validate_feature_names_raises_with_diff_summary() -> None:
    model, _ = _build_tiny_distilbert()

    from defaultplusplus import FeatureExtractor

    fx = FeatureExtractor(model, arch="encoder")
    expected = list(fx.feature_names) + ["my_made_up_column"]
    expected.remove("loss_final")  # also drop one to test "missing" branch

    with pytest.raises(ValueError) as excinfo:
        fx.collector.validate_feature_names(expected)
    msg = str(excinfo.value)
    assert "my_made_up_column" in msg
    assert "loss_final" in msg


# ─────────────────────────────────────────────────────────────────────────
# Encoder vs decoder schemas differ correctly
# ─────────────────────────────────────────────────────────────────────────
def test_encoder_schema_omits_decoder_only_cache_keys() -> None:
    model, _ = _build_tiny_distilbert()

    from defaultplusplus import FeatureExtractor

    fx = FeatureExtractor(model, arch="encoder")
    names = set(fx.feature_names)
    assert "cache_nll_divergence_final" not in names
    assert "cache_hidden_sim_final" not in names


def test_decoder_schema_includes_cache_keys() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    model = AutoModelForCausalLM.from_pretrained(name)

    from defaultplusplus import FeatureExtractor

    fx = FeatureExtractor(model, arch="decoder")
    names = set(fx.feature_names)
    assert "cache_nll_divergence_final" in names
    assert "cache_hidden_sim_final" in names
