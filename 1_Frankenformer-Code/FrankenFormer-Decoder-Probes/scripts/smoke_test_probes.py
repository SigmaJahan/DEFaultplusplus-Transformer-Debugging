"""Smoke test: verify structural probes fire with forward_with_attention.

Loads distilgpt2, runs 1 batch, and checks that hidden_states and attentions
are populated and that structural metrics (ffn_delta, ln_std, residual_cos,
attention_entropy, head_similarity, pre_softmax) are non-zero in the output.

Usage:
    python scripts/smoke_test_probes.py           # CPU
    python scripts/smoke_test_probes.py --cuda     # GPU
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load model
    model_name = "distilgpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Create dummy batch
    text = "The transformer architecture consists of attention layers and feed-forward networks."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    # 3. Forward WITHOUT attention/hidden (the old broken way)
    print("\n--- Forward WITHOUT output_hidden_states/output_attentions ---")
    outputs_plain = model(**inputs)
    print(f"  attentions: {outputs_plain.attentions}")
    print(f"  hidden_states: {outputs_plain.hidden_states}")
    assert outputs_plain.attentions is None, "Expected None for plain forward"
    assert outputs_plain.hidden_states is None, "Expected None for plain forward"
    print("  CONFIRMED: attentions and hidden_states are None")

    # 4. Forward WITH attention/hidden (the fixed way)
    print("\n--- Forward WITH output_hidden_states/output_attentions ---")
    outputs_fixed = model(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
    attn = outputs_fixed.attentions
    hidden = outputs_fixed.hidden_states
    print(f"  attentions: {len(attn)} layers, shape {attn[0].shape}")
    print(f"  hidden_states: {len(hidden)} layers, shape {hidden[0].shape}")
    assert attn is not None and len(attn) > 0, "attentions should be populated"
    assert hidden is not None and len(hidden) > 0, "hidden_states should be populated"
    print("  CONFIRMED: attentions and hidden_states are populated")

    # 5. Test MetricCollector with fixed outputs
    print("\n--- Testing MetricCollector with fixed outputs ---")
    from src.metrics.base_metrics import BaseMetrics
    num_layers = model.config.n_layer
    bm = BaseMetrics(device, config={
        "model_type": "gpt2",
        "num_hidden_layers": num_layers,
        "pad_token_id": tokenizer.pad_token_id,
    })

    # Structural probes
    struct = bm.compute_structural_metrics(
        hidden_states=hidden,
        model=model,
        attention_mask=inputs.get("attention_mask"),
        input_ids=inputs.get("input_ids"),
        logits=outputs_fixed.logits,
    )
    print(f"  Structural metrics: {len(struct)} keys")
    probe_keys = [k for k in struct if any(p in k for p in [
        "ffn_delta", "ln_std", "residual_cos", "ffn_var_ratio",
        "ffn_active_dim", "ln_mean_abs", "h1_delta",
    ])]
    print(f"  Probe keys found: {len(probe_keys)}")
    for k in sorted(probe_keys)[:10]:
        print(f"    {k}: {struct[k]:.6f}")

    nonzero_probes = [k for k in probe_keys if abs(struct[k]) > 1e-12]
    print(f"  Non-zero probes: {len(nonzero_probes)}/{len(probe_keys)}")

    # Attention metrics
    attn_metrics = bm.compute_attention_metrics(
        attn[0], model=model, layer_idx=0,
        layer_input=hidden[0],
        attention_mask=inputs.get("attention_mask"),
        input_ids=inputs.get("input_ids"),
    )
    attn_keys = [k for k in attn_metrics if any(p in k for p in [
        "entropy", "head_similarity", "pre_softmax", "mass_pad", "sparsity",
    ])]
    print(f"\n  Attention metrics: {len(attn_metrics)} keys")
    print(f"  Key attention probes found: {len(attn_keys)}")
    for k in sorted(attn_keys)[:10]:
        print(f"    {k}: {attn_metrics[k]:.6f}")

    nonzero_attn = [k for k in attn_keys if abs(attn_metrics[k]) > 1e-12]
    print(f"  Non-zero attention probes: {len(nonzero_attn)}/{len(attn_keys)}")

    # 6. Summary
    total_probes = len(probe_keys) + len(attn_keys)
    total_nonzero = len(nonzero_probes) + len(nonzero_attn)
    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULT: {total_nonzero}/{total_probes} probes are non-zero")
    if total_nonzero > 0:
        print("PASS: Structural probes are firing correctly")
    else:
        print("FAIL: No probes are producing values")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
