"""Extract DEFault++ features through the HuggingFace ``Trainer``.

Drop ``DEFaultPlusCallback`` into ``Trainer(callbacks=...)``. The
callback enables ``output_attentions`` / ``output_hidden_states`` on
the model's config so attention metrics fire, captures inputs / outputs
through the trainer's compute_loss override, and writes the final
feature vector to disk on ``on_train_end``.

Run:

    python examples/extract_with_hf_trainer.py
"""
from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from defaultplusplus.hf_callback import DEFaultPlusCallback


class CallbackAwareTrainer(Trainer):
    """Trainer subclass that hands inputs / outputs to the callback.

    HuggingFace's default trainer does not pass the batch inputs or
    model outputs to ``on_step_end``. Override ``compute_loss`` so the
    callback captures both, which lets the per-step metrics include
    attention weights and hidden states.
    """

    def __init__(self, *args, defaultpp_callback: DEFaultPlusCallback, **kwargs):
        super().__init__(*args, **kwargs)
        self._defaultpp_callback = defaultpp_callback

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        # The callback only needs a reference; do not detach so HF
        # gradient flow remains unchanged.
        self._defaultpp_callback.capture_inputs(dict(inputs))
        outputs = model(**inputs)
        self._defaultpp_callback.capture_outputs(outputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    model_name = "distilbert-base-uncased"
    out_path = Path("features.json")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Synthetic SST-2-style dataset.
    examples = [
        ("the movie was great", 1),
        ("loved the cinematography", 1),
        ("the movie was bad", 0),
        ("a tedious slog", 0),
    ] * 16
    ds = Dataset.from_dict({"text": [e[0] for e in examples],
                            "label": [e[1] for e in examples]})
    ds = ds.map(lambda x: tok(x["text"], padding="max_length",
                              truncation=True, max_length=32), batched=True)
    ds = ds.rename_column("label", "labels")
    ds = ds.with_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir="/tmp/defaultpp_hf_demo",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        logging_steps=4,
        save_strategy="no",
        report_to=[],
    )

    callback = DEFaultPlusCallback(out_path=out_path, arch="encoder")
    import inspect
    from transformers import Trainer as _BaseTrainer
    tok_kwarg = ("processing_class"
                 if "processing_class" in inspect.signature(_BaseTrainer.__init__).parameters
                 else "tokenizer")
    trainer = CallbackAwareTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        callbacks=[callback],
        defaultpp_callback=callback,
        **{tok_kwarg: tok},
    )
    trainer.train()

    feature_vector = callback.feature_vector or {}
    print(f"[example] {len(feature_vector)} features written to {out_path}")


if __name__ == "__main__":
    main()
