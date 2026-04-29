"""Extract DEFault++ features from a manual fine-tuning loop.

This example runs a few training steps on a tiny DistilBERT model with
synthetic SST-2-style inputs and writes the resulting feature vector
to ``features.json``. Use it as a template for instrumenting your own
training loop.

Run:

    python examples/extract_during_finetune.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from defaultplusplus import FeatureExtractor


def main() -> None:
    model_name = "distilbert-base-uncased"
    print(f"[example] loading {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        output_attentions=True, output_hidden_states=True)
    model.train()

    # Synthetic batches.
    batches = []
    for _ in range(8):
        text = ["the movie was great"] * 8 + ["the movie was bad"] * 8
        labels = torch.tensor([1] * 8 + [0] * 8)
        enc = tok(text, padding=True, truncation=True, return_tensors="pt")
        enc["labels"] = labels
        batches.append(enc)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    out_path = Path("features.json")
    print("[example] starting fine-tune with FeatureExtractor")

    with FeatureExtractor(model, arch="encoder") as fx:
        for epoch in range(2):
            for step, batch in enumerate(batches):
                t0 = time.perf_counter()
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                fx.step(
                    loss=outputs.loss,
                    outputs=outputs,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    optimizer=optimizer,
                    step_time=time.perf_counter() - t0,
                )
            fx.epoch_end(epoch)
            # Pretend the eval loop returned these:
            fx.record_validation(epoch, {"accuracy": 0.85, "loss": 0.42})

        feature_vector = fx.finalize()
        fx.to_json(out_path)

    print(f"[example] {len(feature_vector)} features written to {out_path}")
    print("[example] sample keys:")
    for k in sorted(feature_vector.keys())[:10]:
        print(f"  {k} = {feature_vector[k]:.4f}")


if __name__ == "__main__":
    main()
