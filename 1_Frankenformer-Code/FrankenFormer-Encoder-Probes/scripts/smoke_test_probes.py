#!/usr/bin/env python3
"""Unit test for structural and attention probes."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_encoder_model
from src.metrics.metric_collector import MetricCollector


def test_structural_probes():
    """Load distilbert, forward with hidden_states + attentions, validate probe metrics."""
    print("=== Smoke Test: Structural Probes ===\n")
    device = torch.device("cpu")

    print("Loading distilbert-base-uncased...")
    model_wrapper = load_encoder_model("distilbert-base-uncased", num_labels=2, device=device)
    tokenizer = model_wrapper.tokenizer
    num_layers = model_wrapper.get_num_layers()
    print(f"  Layers: {num_layers}, Hidden: {model_wrapper.get_hidden_size()}")

    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Forward pass with hidden_states + attentions...")
    outputs = model_wrapper.forward_with_attention(**inputs)
    assert outputs.hidden_states is not None, "hidden_states is None"
    assert outputs.attentions is not None, "attentions is None"
    assert len(outputs.hidden_states) == num_layers + 1, \
        f"Expected {num_layers+1} hidden states, got {len(outputs.hidden_states)}"
    assert len(outputs.attentions) == num_layers, \
        f"Expected {num_layers} attention layers, got {len(outputs.attentions)}"
    print(f"  hidden_states: {len(outputs.hidden_states)} tensors, shape={outputs.hidden_states[0].shape}")
    print(f"  attentions: {len(outputs.attentions)} tensors, shape={outputs.attentions[0].shape}")

    print("\nTesting MetricCollector with fixed outputs...")
    collector = MetricCollector(
        device=device,
        collect_per_batch=True,
        collect_attention=True,
        config={
            "num_hidden_layers": num_layers,
            "activation_interval": 1,
            "gradient_window": 5,
        },
    )

    labels = torch.zeros(1, dtype=torch.long, device=device)
    metrics = collector.collect_batch_metrics(
        loss=0.693,
        model=model_wrapper.model,
        optimizer=None,
        outputs=outputs,
        labels=labels,
        batch_idx=0,
        epoch=0,
        batch=inputs,
    )

    print(f"  Total metrics collected: {len(metrics)}")

    print("\nValidating structural probe metrics...")
    structural_checks = {
        "ffn_delta": [k for k in metrics if "ffn_delta" in k],
        "ln_std": [k for k in metrics if "ln_std" in k],
        "residual_cos": [k for k in metrics if "residual_cos" in k],
    }

    all_pass = True
    for probe_name, keys in structural_checks.items():
        if not keys:
            print(f"  WARN: No keys found for {probe_name}")
            all_pass = False
            continue
        nonzero = [k for k in keys if abs(metrics[k]) > 1e-12]
        if nonzero:
            sample_key = nonzero[0]
            print(f"  PASS {probe_name}: {len(keys)} keys, e.g. {sample_key}={metrics[sample_key]:.6f}")
        else:
            print(f"  FAIL {probe_name}: all {len(keys)} values are zero")
            all_pass = False

    print("\nValidating attention probe metrics...")
    attention_checks = {
        "entropy": [k for k in metrics if "entropy" in k.lower() and "attention" in k.lower()],
        "head_similarity": [k for k in metrics if "head_similarity" in k],
        "pre_softmax": [k for k in metrics if "pre_softmax" in k],
    }

    for probe_name, keys in attention_checks.items():
        if not keys:
            print(f"  WARN: No keys found for {probe_name}")
            continue
        nonzero = [k for k in keys if abs(metrics[k]) > 1e-12]
        if nonzero:
            sample_key = nonzero[0]
            print(f"  PASS {probe_name}: {len(keys)} keys, e.g. {sample_key}={metrics[sample_key]:.6f}")
        else:
            print(f"  FAIL {probe_name}: all {len(keys)} values are zero")
            all_pass = False

    del model_wrapper
    print(f"\n{'PASS' if all_pass else 'FAIL'}: Smoke test {'passed' if all_pass else 'had failures'}")
    return all_pass


if __name__ == "__main__":
    success = test_structural_probes()
    sys.exit(0 if success else 1)
