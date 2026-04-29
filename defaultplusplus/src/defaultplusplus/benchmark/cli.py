"""Command-line benchmark driver for DEFault-bench construction."""
from __future__ import annotations

import argparse
import inspect
import math
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from ..deform import FaultInjector, get_injector
from ..deform.fault_config import FaultConfiguration
from ..deform.operators import OPERATORS
from .config_grid import BenchmarkSpec, enumerate_configurations
from .dataset_writer import DatasetWriter
from .runner import run_one_configuration


_GLUE_TASKS = {"sst2", "qnli", "rte", "mrpc", "qqp"}


class _CallbackAwareTrainer:
    """Mixin factory for capturing inputs and outputs in HF Trainer."""

    @staticmethod
    def build(base_cls, callback):
        class CallbackAwareTrainer(base_cls):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
                callback.capture_inputs(dict(inputs))
                outputs = model(**inputs)
                callback.capture_outputs(outputs)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

        return CallbackAwareTrainer


class HFFineTuneFn:
    """Default HuggingFace Trainer fine-tuning function used by the CLI."""

    def __init__(self,
                 *,
                 arch: str,
                 epochs: float,
                 batch_size: int,
                 learning_rate: float,
                 max_train_samples: int,
                 max_eval_samples: int,
                 synthetic_fallback: bool,
                 work_dir: Path) -> None:
        self.arch = arch
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.synthetic_fallback = synthetic_fallback
        self.work_dir = work_dir

    def __call__(self,
                 model_name: str,
                 task: str,
                 seed: int,
                 injector: FaultInjector | Any | None,
                 ) -> tuple[float, dict[str, float]]:
        _seed_everything(seed)

        try:
            if self.arch == "encoder":
                return self._run_encoder(model_name, task, seed, injector)
            return self._run_decoder(model_name, task, seed, injector)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_encoder(self,
                     model_name: str,
                     task: str,
                     seed: int,
                     injector: FaultInjector | Any | None,
                     ) -> tuple[float, dict[str, float]]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds, eval_ds, num_labels = _load_encoder_dataset(
            task, tokenizer, self.max_train_samples, self.max_eval_samples,
            synthetic_fallback=self.synthetic_fallback,
        )
        model = _from_pretrained_eager_attention(
            AutoModelForSequenceClassification, model_name, num_labels=num_labels
        )
        return self._train_with_trainer(
            model=model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            seed=seed,
            injector=injector,
            metric_name="accuracy",
            higher_is_better=True,
        )

    def _run_decoder(self,
                     model_name: str,
                     task: str,
                     seed: int,
                     injector: FaultInjector | Any | None,
                     ) -> tuple[float, dict[str, float]]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_ds, eval_ds = _load_decoder_dataset(
            task, tokenizer, self.max_train_samples, self.max_eval_samples,
            synthetic_fallback=self.synthetic_fallback,
        )
        model = _from_pretrained_eager_attention(AutoModelForCausalLM, model_name)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return self._train_with_trainer(
            model=model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            seed=seed,
            injector=injector,
            metric_name="eval_loss",
            higher_is_better=False,
        )

    def _train_with_trainer(self,
                            *,
                            model: nn.Module,
                            tokenizer: Any,
                            train_ds: Any,
                            eval_ds: Any,
                            seed: int,
                            injector: FaultInjector | Any | None,
                            metric_name: str,
                            higher_is_better: bool,
                            ) -> tuple[float, dict[str, float]]:
        from transformers import Trainer, TrainingArguments
        from defaultplusplus.hf_callback import DEFaultPlusCallback

        callback = DEFaultPlusCallback(arch=self.arch)
        trainer_cls = _CallbackAwareTrainer.build(Trainer, callback)
        output_dir = self.work_dir / f"seed_{seed}_{random.randrange(10**9)}"
        args = _training_args(
            output_dir=output_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=seed,
        )

        kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "callbacks": [callback],
        }
        if self.arch == "encoder":
            kwargs["compute_metrics"] = _accuracy_metrics

        tok_kwarg = ("processing_class"
                     if "processing_class" in inspect.signature(Trainer.__init__).parameters
                     else "tokenizer")
        kwargs[tok_kwarg] = tokenizer

        trainer = trainer_cls(**kwargs)
        bound_injector = _materialize_injector(injector, model)
        context = bound_injector if bound_injector is not None else nullcontext()
        with context:
            trainer.train()
            metrics = trainer.evaluate()

        feature_vector = callback.feature_vector
        if feature_vector is None and callback.extractor is not None:
            feature_vector = callback.extractor.finalize()
        feature_vector = dict(feature_vector or {})

        metric = _select_metric(metrics, metric_name, higher_is_better)
        return metric, feature_vector


def default_feature_builder(clean: list[dict[str, float]],
                            faulty: list[dict[str, float]]) -> dict[str, float]:
    """Average paired feature dictionaries and return faulty-minus-clean deltas."""
    keys = sorted(set().union(*(d.keys() for d in clean + faulty)))
    out = {}
    for key in keys:
        c = _mean([d.get(key, 0.0) for d in clean])
        f = _mean([d.get(key, 0.0) for d in faulty])
        out[key] = f - c
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("encoder", "decoder"))
    parser.add_argument("--models", required=True, help="Comma-separated HF model names.")
    parser.add_argument("--tasks", required=True, help="Comma-separated benchmark task names.")
    parser.add_argument("--operators", default="", help="Comma-separated operator IDs; empty means all.")
    parser.add_argument("--severities", default="low,medium,high")
    parser.add_argument("--seeds", default="42,123,456,789,101112")
    parser.add_argument("--layers", default="",
                        help="Comma-separated 1-indexed layers for one mutant; empty means natural target.")
    parser.add_argument("--output", default="",
                        help="CSV path. Default: data/{arch}_benchmark.csv")
    parser.add_argument("--status-dir", default="",
                        help="Optional directory for per-config status JSON files.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-synthetic-fallback", action="store_true",
                        help="Fail instead of using a tiny synthetic dataset when HF datasets are unavailable.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    seeds = _parse_ints(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    operators = _parse_csv(args.operators)
    _validate_operators(operators)
    output = Path(args.output) if args.output else Path("data") / f"{args.arch}_benchmark.csv"

    spec = BenchmarkSpec(
        arch=args.arch,
        models=_parse_csv(args.models),
        tasks=_parse_csv(args.tasks),
        operators=operators,
        layer_sets=(_parse_layers(args.layers),),
        severities=_parse_csv(args.severities),
        seeds=(seeds[0],),
    )

    work_dir = Path(tempfile.mkdtemp(prefix="defaultpp_benchmark_"))
    fine_tune = HFFineTuneFn(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        synthetic_fallback=not args.no_synthetic_fallback,
        work_dir=work_dir,
    )

    outcomes = []
    for config in enumerate_configurations(spec):
        outcome = run_one_configuration(
            config=config,
            injector_factory=_lazy_injector_factory,
            fine_tune=fine_tune,
            feature_builder=default_feature_builder,
            higher_is_better=(args.arch == "encoder"),
            alpha=args.alpha,
            seeds=seeds,
            output_dir=Path(args.status_dir) if args.status_dir else None,
        )
        outcomes.append(outcome)
        print(f"[defaultpp-benchmark] {config.config_id()} killed={outcome.mutant.killed} "
              f"p={outcome.mutant.p_value:.5g}")

    feature_columns = sorted(set().union(*(o.mutant.feature_vector.keys() for o in outcomes)))
    writer = DatasetWriter(output, fixed_columns=feature_columns)
    for outcome in outcomes:
        writer.append(outcome.mutant)

    print(f"[defaultpp-benchmark] wrote {len(outcomes)} row(s) to {output}")
    return 0


def _lazy_injector_factory(config: FaultConfiguration):
    injector_cls = get_injector(
        config.operator_id,
        layers=config.layers,
        param_value=config.param_value,
        severity=config.severity,
    )

    def bind(model: nn.Module) -> FaultInjector:
        return injector_cls(model)

    return bind


def _materialize_injector(injector: Any, model: nn.Module) -> FaultInjector | None:
    if injector is None:
        return None
    if isinstance(injector, FaultInjector):
        return injector
    if callable(injector):
        bound = injector(model)
        if not isinstance(bound, FaultInjector):
            raise TypeError("injector factory must return a FaultInjector")
        return bound
    raise TypeError("injector must be a FaultInjector or callable")


def _training_args(**kwargs):
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return TrainingArguments(
        output_dir=str(kwargs["output_dir"]),
        num_train_epochs=kwargs["epochs"],
        per_device_train_batch_size=kwargs["batch_size"],
        per_device_eval_batch_size=kwargs["batch_size"],
        learning_rate=kwargs["learning_rate"],
        seed=kwargs["seed"],
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        **{eval_key: "epoch"},
    )


def _from_pretrained_eager_attention(model_cls: Any, model_name: str, **kwargs: Any):
    try:
        return model_cls.from_pretrained(
            model_name, attn_implementation="eager", **kwargs
        )
    except TypeError:
        return model_cls.from_pretrained(model_name, **kwargs)


def _load_encoder_dataset(task: str,
                          tokenizer: Any,
                          max_train: int,
                          max_eval: int,
                          *,
                          synthetic_fallback: bool):
    if task in _GLUE_TASKS:
        try:
            from datasets import load_dataset

            raw = load_dataset("glue", task)
            text_cols = _glue_text_columns(task)

            def tokenize(batch):
                if len(text_cols) == 1:
                    return tokenizer(batch[text_cols[0]], padding="max_length",
                                     truncation=True, max_length=64)
                return tokenizer(batch[text_cols[0]], batch[text_cols[1]],
                                 padding="max_length", truncation=True, max_length=64)

            train = raw["train"].select(range(min(max_train, len(raw["train"]))))
            eval_name = "validation_matched" if task == "mnli" else "validation"
            eval_ds = raw[eval_name].select(range(min(max_eval, len(raw[eval_name]))))
            train = train.map(tokenize, batched=True).rename_column("label", "labels")
            eval_ds = eval_ds.map(tokenize, batched=True).rename_column("label", "labels")
            cols = ["input_ids", "attention_mask", "labels"]
            return train.with_format("torch", columns=cols), eval_ds.with_format("torch", columns=cols), 2
        except Exception:
            if not synthetic_fallback:
                raise

    if not synthetic_fallback:
        raise ValueError(f"No built-in loader for encoder task {task!r}")
    return _synthetic_encoder_dataset(tokenizer, max_train, max_eval)


def _load_decoder_dataset(task: str,
                          tokenizer: Any,
                          max_train: int,
                          max_eval: int,
                          *,
                          synthetic_fallback: bool):
    try:
        from datasets import load_dataset

        if task == "wikitext2":
            raw = load_dataset("wikitext", "wikitext-2-raw-v1")
            train_text = [x for x in raw["train"]["text"] if x.strip()][:max_train]
            eval_text = [x for x in raw["validation"]["text"] if x.strip()][:max_eval]
            return (_causal_lm_dataset(tokenizer, train_text),
                    _causal_lm_dataset(tokenizer, eval_text))
    except Exception:
        if not synthetic_fallback:
            raise

    if not synthetic_fallback:
        raise ValueError(f"No built-in loader for decoder task {task!r}")
    text = [
        "language models predict the next token",
        "attention stores previous keys and values",
        "transformer debugging requires paired traces",
        "fault injection changes training behavior",
    ]
    return (_causal_lm_dataset(tokenizer, (text * max(1, max_train // len(text)))[:max_train]),
            _causal_lm_dataset(tokenizer, (text * max(1, max_eval // len(text)))[:max_eval]))


def _synthetic_encoder_dataset(tokenizer: Any, max_train: int, max_eval: int):
    examples = [
        ("the movie was clear and engaging", 1),
        ("the argument was coherent", 1),
        ("the movie was dull and confused", 0),
        ("the explanation was incoherent", 0),
    ]

    def build(n: int):
        rows = (examples * max(1, math.ceil(n / len(examples))))[:n]
        enc = tokenizer([r[0] for r in rows], padding="max_length",
                        truncation=True, max_length=64, return_tensors="pt")
        enc["labels"] = torch.tensor([r[1] for r in rows], dtype=torch.long)
        return _TorchDictDataset(enc)

    return build(max_train), build(max_eval), 2


def _causal_lm_dataset(tokenizer: Any, texts: list[str]):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=64, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return _TorchDictDataset(enc)


class _TorchDictDataset(torch.utils.data.Dataset):
    def __init__(self, columns: dict[str, torch.Tensor]) -> None:
        self.columns = columns
        self.length = len(next(iter(columns.values()))) if columns else 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.columns.items()}


def _glue_text_columns(task: str) -> tuple[str, ...]:
    return {
        "sst2": ("sentence",),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
    }[task]


def _accuracy_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def _select_metric(metrics: dict[str, Any], name: str, higher_is_better: bool) -> float:
    if name in metrics:
        return float(metrics[name])
    eval_name = f"eval_{name}"
    if eval_name in metrics:
        return float(metrics[eval_name])
    if "eval_accuracy" in metrics:
        return float(metrics["eval_accuracy"])
    if "eval_loss" in metrics:
        loss = float(metrics["eval_loss"])
        return -loss if higher_is_better else loss
    return 0.0


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_layers(value: str) -> tuple[int, ...]:
    return _parse_ints(value)


def _validate_operators(operators: tuple[str, ...]) -> None:
    unknown = [op for op in operators if op not in OPERATORS]
    if unknown:
        raise SystemExit(f"Unknown operator IDs: {', '.join(unknown)}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: Iterable[Any]) -> float:
    nums = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            f = 0.0
        nums.append(f if math.isfinite(f) else 0.0)
    return sum(nums) / len(nums) if nums else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
