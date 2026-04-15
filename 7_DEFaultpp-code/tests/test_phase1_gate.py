"""Phase 1 gate tests — all must pass before proceeding to Phase 2/3."""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM

from src.defaultplusplus.extraction.inspector import ModelInspector
from src.defaultplusplus.extraction.metrics.training import TrainingMetrics
from src.defaultplusplus.extraction.metrics.gradient import GradientMetrics
from src.defaultplusplus.extraction.metrics.attention import AttentionMetrics
from src.defaultplusplus.extraction.metrics.structural import StructuralMetrics
from src.defaultplusplus.extraction.metrics.logit import LogitMetrics
from src.defaultplusplus.extraction.metrics.positional import PositionalMetrics
from src.defaultplusplus.extraction.metrics.cache import CacheMetrics
from src.defaultplusplus.extraction.collector import MetricCollector
from src.defaultplusplus.extraction.aggregator import (
    OnlineStatistic, EpochAggregator, compute_window_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bert_model():
    return AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-BertForSequenceClassification",
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def gpt2_model():
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2",
        attn_implementation="eager",
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model


@pytest.fixture(scope="module")
def bert_inspector(bert_model):
    return ModelInspector(bert_model)


@pytest.fixture(scope="module")
def gpt2_inspector(gpt2_model):
    return ModelInspector(gpt2_model)


def _run_bert_forward(model):
    """Run a forward+backward pass on the BERT model, return outputs."""
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 2, (batch_size,))

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    return outputs, input_ids, attention_mask, labels


def _run_gpt2_forward(model):
    """Run a forward+backward pass on the GPT-2 model, return outputs."""
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    return outputs, input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# T1.1 – BERT-style category detection
# ---------------------------------------------------------------------------
class TestInspectorEncoder:
    def test_arch_family(self, bert_inspector):
        assert bert_inspector.arch_family == "encoder"

    def test_layers_discovered(self, bert_inspector):
        assert bert_inspector.layers is not None
        assert len(bert_inspector.layers) > 0

    def test_attention_discovered(self, bert_inspector):
        assert bert_inspector._attn_pattern is not None
        assert bert_inspector._attn_pattern.module_attr != ""

    def test_ffn_discovered(self, bert_inspector):
        assert bert_inspector._ffn_pattern is not None

    def test_layernorm_discovered(self, bert_inspector):
        assert len(bert_inspector._ln_names) > 0


# ---------------------------------------------------------------------------
# T1.2 – GPT-style category detection
# ---------------------------------------------------------------------------
class TestInspectorDecoder:
    def test_arch_family(self, gpt2_inspector):
        assert gpt2_inspector.arch_family == "decoder"

    def test_layers_discovered(self, gpt2_inspector):
        assert gpt2_inspector.layers is not None
        assert len(gpt2_inspector.layers) > 0

    def test_attention_discovered(self, gpt2_inspector):
        assert gpt2_inspector._attn_pattern is not None

    def test_mlp_discovered(self, gpt2_inspector):
        assert gpt2_inspector._ffn_pattern is not None
        assert gpt2_inspector._ffn_pattern.module_attr != ""

    def test_layernorm_discovered(self, gpt2_inspector):
        assert len(gpt2_inspector._ln_names) > 0


# ---------------------------------------------------------------------------
# T1.3 – Variant within same category
# ---------------------------------------------------------------------------
def test_distilbert_resolves_as_encoder():
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
        attn_implementation="eager",
    )
    inspector = ModelInspector(model)
    assert inspector.arch_family == "encoder"
    assert len(inspector.layers) > 0


# ---------------------------------------------------------------------------
# T1.4 – Attention module count matches num_layers
# ---------------------------------------------------------------------------
def test_attention_module_count(bert_inspector):
    count = sum(
        1 for i in range(len(bert_inspector.layers))
        if bert_inspector.get_attention_module(i) is not None
    )
    assert count == len(bert_inspector.layers)


# ---------------------------------------------------------------------------
# T1.5 – Embedding discovery
# ---------------------------------------------------------------------------
def test_embedding_is_embedding(bert_inspector):
    assert isinstance(bert_inspector.embedding, nn.Embedding)


# ---------------------------------------------------------------------------
# T1.6 – Parameter groups
# ---------------------------------------------------------------------------
def test_parameter_groups(bert_inspector):
    groups = bert_inspector.get_parameter_groups()
    assert "embedding" in groups
    assert "layer0_attention" in groups
    assert "layer0_ffn" in groups


# ---------------------------------------------------------------------------
# T1.7 – Unsupported model
# ---------------------------------------------------------------------------
def test_unsupported_model_raises():
    """A bare nn.Module with no config should raise ValueError."""
    model = nn.Linear(10, 10)
    with pytest.raises(ValueError):
        ModelInspector(model)


# ---------------------------------------------------------------------------
# T1.7b – Unknown-but-compatible model
# ---------------------------------------------------------------------------
def test_unknown_compatible_model():
    """Custom nn.Module with BERT-like structure should be discoverable."""
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, vocab_size=100, num_labels=2,
    )
    model = BertForSequenceClassification(config)
    inspector = ModelInspector(model)
    assert inspector.arch_family == "encoder"
    assert inspector.num_layers == 2
    assert inspector.embedding is not None


# ---------------------------------------------------------------------------
# T1.8 – TrainingMetrics
# ---------------------------------------------------------------------------
def test_training_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    module = TrainingMetrics(bert_inspector)
    result = module.collect(
        loss=outputs.loss, model=bert_model, optimizer=optimizer, step_time=0.1,
    )
    assert "train_loss" in result
    assert "loss" in result
    assert isinstance(result["train_loss"], float)


# ---------------------------------------------------------------------------
# T1.9 – GradientMetrics
# ---------------------------------------------------------------------------
def test_gradient_metrics(bert_model, bert_inspector):
    _run_bert_forward(bert_model)
    module = GradientMetrics(bert_inspector)
    result = module.collect(model=bert_model)
    assert "grad_norm_total" in result
    assert not math.isnan(result["grad_norm_total"])


# ---------------------------------------------------------------------------
# T1.10 – AttentionMetrics
# ---------------------------------------------------------------------------
def test_attention_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    module = AttentionMetrics(bert_inspector)
    result = module.collect(
        model=bert_model,
        attention_weights=outputs.attentions,
        hidden_states=outputs.hidden_states,
        attention_mask=attention_mask,
        input_ids=input_ids,
    )
    assert "attention_entropy_mean" in result


# ---------------------------------------------------------------------------
# T1.11 – StructuralMetrics
# ---------------------------------------------------------------------------
def test_structural_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    module = StructuralMetrics(bert_inspector)
    result = module.collect(
        model=bert_model,
        hidden_states=outputs.hidden_states,
        input_ids=input_ids,
    )
    assert "ffn_delta_mean" in result


# ---------------------------------------------------------------------------
# T1.12 – LogitMetrics
# ---------------------------------------------------------------------------
def test_logit_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    module = LogitMetrics(bert_inspector)
    result = module.collect(outputs=outputs, labels=labels)
    assert "accuracy" in result
    assert "logit_entropy" in result


# ---------------------------------------------------------------------------
# T1.13 – PositionalMetrics
# ---------------------------------------------------------------------------
def test_positional_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    module = PositionalMetrics(bert_inspector)
    result = module.collect(
        model=bert_model, outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    assert "positional_accuracy_early" in result


# ---------------------------------------------------------------------------
# T1.14 – CacheMetrics decoder-only
# ---------------------------------------------------------------------------
def test_cache_metrics_decoder_only(bert_inspector, gpt2_inspector):
    # Cache should not be in encoder modules
    encoder_modules = [type(m).__name__ for m in [
        TrainingMetrics(bert_inspector), GradientMetrics(bert_inspector),
    ]]
    # Just verify the module can be instantiated for decoder
    cache = CacheMetrics(gpt2_inspector)
    assert cache is not None
    assert gpt2_inspector.arch_family == "decoder"


# ---------------------------------------------------------------------------
# T1.15 – No NaN under normal input
# ---------------------------------------------------------------------------
def test_no_nan_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    collector = MetricCollector(bert_inspector)
    result = collector.collect_step(
        loss=outputs.loss, model=bert_model, optimizer=optimizer,
        outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    for key, value in result.items():
        if isinstance(value, float):
            assert not math.isnan(value), f"{key} is NaN"


# ---------------------------------------------------------------------------
# T1.16 – MetricCollector returns >50 metrics
# ---------------------------------------------------------------------------
def test_collector_many_metrics(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    collector = MetricCollector(bert_inspector)
    result = collector.collect_step(
        loss=outputs.loss, model=bert_model, optimizer=optimizer,
        outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    float_keys = [k for k, v in result.items() if isinstance(v, float)]
    assert len(float_keys) >= 50, f"Only {len(float_keys)} float metrics"


# ---------------------------------------------------------------------------
# T1.17 – finalize_epoch keys
# ---------------------------------------------------------------------------
def test_finalize_epoch_keys(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    collector = MetricCollector(bert_inspector)
    collector.collect_step(
        loss=outputs.loss, model=bert_model, optimizer=optimizer,
        outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    epoch_metrics = collector.finalize_epoch(0)
    mean_keys = [k for k in epoch_metrics if k.endswith("_mean")]
    var_keys = [k for k in epoch_metrics if k.endswith("_var")]
    assert len(mean_keys) > 0
    assert len(var_keys) > 0


# ---------------------------------------------------------------------------
# T1.18 – get_final_features
# ---------------------------------------------------------------------------
def test_get_final_features(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    collector = MetricCollector(bert_inspector)
    collector.collect_step(
        loss=outputs.loss, model=bert_model, optimizer=optimizer,
        outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    collector.finalize_epoch(0)
    final = collector.get_final_features()
    assert isinstance(final, dict)
    assert len(final) > 0


# ---------------------------------------------------------------------------
# T1.19 – feature_names deterministic
# ---------------------------------------------------------------------------
def test_feature_names_deterministic(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    collector = MetricCollector(bert_inspector)
    collector.collect_step(
        loss=outputs.loss, model=bert_model, optimizer=optimizer,
        outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    collector.finalize_epoch(0)
    names1 = collector.feature_names
    names2 = collector.feature_names
    names3 = collector.feature_names
    assert names1 == names2 == names3


# ---------------------------------------------------------------------------
# T1.20 – No duplicate feature names
# ---------------------------------------------------------------------------
def test_no_duplicate_feature_names(bert_model, bert_inspector):
    outputs, input_ids, attention_mask, labels = _run_bert_forward(bert_model)
    optimizer = torch.optim.SGD(bert_model.parameters(), lr=0.01)
    collector = MetricCollector(bert_inspector)
    collector.collect_step(
        loss=outputs.loss, model=bert_model, optimizer=optimizer,
        outputs=outputs, labels=labels,
        attention_mask=attention_mask, input_ids=input_ids,
    )
    collector.finalize_epoch(0)
    names = collector.feature_names
    assert len(names) == len(set(names)), "Duplicate feature names found"


# ---------------------------------------------------------------------------
# T1.21 – OnlineStatistic matches numpy
# ---------------------------------------------------------------------------
def test_online_statistic_matches_numpy():
    rng = np.random.default_rng(42)
    values = rng.standard_normal(100)
    stat = OnlineStatistic()
    for v in values:
        stat.update(float(v))
    assert abs(stat.mean - np.mean(values)) < 1e-10
    assert abs(stat.variance - np.var(values, ddof=1)) < 1e-10


# ---------------------------------------------------------------------------
# T1.22 – EpochAggregator
# ---------------------------------------------------------------------------
def test_epoch_aggregator():
    agg = EpochAggregator()
    agg.update({"loss": 1.0, "acc": 0.5})
    agg.update({"loss": 0.5, "acc": 0.8})
    result = agg.finalize_epoch(0)
    assert "loss_mean" in result
    assert "loss_var" in result
    assert "acc_mean" in result
    assert abs(result["loss_mean"] - 0.75) < 1e-10


# ---------------------------------------------------------------------------
# T1.23 – compute_window_features
# ---------------------------------------------------------------------------
def test_compute_window_features():
    history = {
        "loss": [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7), (5, 0.6),
                 (6, 0.5), (7, 0.4), (8, 0.3), (9, 0.2), (10, 0.1)],
    }
    features = compute_window_features(history, total_epochs=10)
    assert "loss_early_mean" in features
    assert "loss_mid_mean" in features
    assert "loss_late_mean" in features
    assert "loss_final" in features
    assert abs(features["loss_final"] - 0.1) < 1e-10
