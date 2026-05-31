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

from ..deform import (
    FaultInjector,
    generate_clean_variants,
    get_injector,
    run_one_clean_variant,
)
from ..deform.fault_config import FaultConfiguration
from ..deform.operators import OPERATORS
from .config_grid import BenchmarkSpec, enumerate_configurations
from .dataset_writer import DatasetWriter
from .runner import run_one_configuration
from .task_metrics import (
    build_compute_metrics, get_task_spec, score_evaluation, supported_tasks,
)


_GLUE_TASKS = {"sst2", "qnli", "rte", "mrpc", "qqp", "stsb", "cola", "mnli"}


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

        spec = get_task_spec(task)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_ds, eval_ds, num_labels = _load_encoder_dataset(
            task, tokenizer, self.max_train_samples, self.max_eval_samples,
            synthetic_fallback=self.synthetic_fallback,
        )
        model_kwargs: dict[str, Any] = {"num_labels": num_labels}
        # STS-B / regression: HF wants problem_type set so the model
        # uses MSE loss instead of cross-entropy on a 1-logit head.
        if "pearson" in spec.raw_metrics or "spearmanr" in spec.raw_metrics:
            model_kwargs["problem_type"] = "regression"
        model = _from_pretrained_eager_attention(
            AutoModelForSequenceClassification, model_name, **model_kwargs,
        )
        return self._train_with_trainer(
            model=model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            seed=seed,
            injector=injector,
            task=task,
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
            task=task,
        )

    def _train_with_trainer(self,
                            *,
                            model: nn.Module,
                            tokenizer: Any,
                            train_ds: Any,
                            eval_ds: Any,
                            seed: int,
                            injector: FaultInjector | Any | None,
                            task: str,
                            ) -> tuple[float, dict[str, float]]:
        from transformers import Trainer, TrainingArguments
        from defaultplusplus.hf_callback import DEFaultPlusCallback

        spec = get_task_spec(task)
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
        # Encoder tasks emit classification or regression metrics that
        # the registry's compute_metrics callable derives from logits +
        # labels. Decoder tasks rely on the loss HF returns by default,
        # so we don't pass compute_metrics.
        if self.arch == "encoder":
            kwargs["compute_metrics"] = build_compute_metrics(task)

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

        try:
            metric = score_evaluation(task, metrics)
        except KeyError as exc:
            # Re-raise with the operative task and the keys that *did*
            # show up so the failure is debuggable in the discard log.
            raise RuntimeError(
                f"task {task!r} aggregator could not score evaluate() output: "
                f"{exc}. Eval keys: {sorted(metrics)}"
            ) from exc
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


def make_clean_fine_tune(base_fine_tune: HFFineTuneFn):
    """Adapt an :class:`HFFineTuneFn` to the clean-variant call signature.

    The faulty path calls ``fine_tune(model, task, seed, injector)``. The
    correct path calls ``fine_tune(model, task, seed, hyperparams)`` with no
    injector and a per-variant hyperparameter override. This adapter applies
    the override (``learning_rate``, ``batch_size``, ``epochs``,
    ``warmup_ratio`` when supported) by cloning the base fine-tune function
    with patched attributes, then runs it with ``injector=None``.
    """
    import copy as _copy

    def _clean_fine_tune(model: str, task: str, seed: int,
                         hyperparams: dict[str, Any]) -> tuple[float, dict]:
        if hyperparams:
            fn = _copy.copy(base_fine_tune)
            for key, value in hyperparams.items():
                # Only patch attributes the fine-tune function actually
                # carries; unknown keys (e.g. warmup_ratio when the trainer
                # does not expose it) are ignored rather than failing.
                if hasattr(fn, key):
                    setattr(fn, key, value)
        else:
            fn = base_fine_tune
        return fn(model, task, seed, None)

    return _clean_fine_tune


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("encoder", "decoder"))
    parser.add_argument("--models", required=True, help="Comma-separated HF model names.")
    parser.add_argument(
        "--tasks", required=True,
        help=("Comma-separated benchmark task names. Supported: "
              + ", ".join(supported_tasks())),
    )
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
    parser.add_argument("--clean-variants", type=int, default=0,
                        help=("Number of label-preserving clean variants to "
                              "generate per (model, task) for the correct "
                              "class. Each variant is tested against the base "
                              "model with the same kill test; variants that "
                              "stay indistinguishable are kept as correct "
                              "samples. 0 disables correct-class generation."))
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
    tasks = _parse_csv(args.tasks)
    _validate_tasks(tasks, arch=args.arch)
    output = Path(args.output) if args.output else Path("data") / f"{args.arch}_benchmark.csv"

    spec = BenchmarkSpec(
        arch=args.arch,
        models=_parse_csv(args.models),
        tasks=tasks,
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

    verifier_factory = _build_verifier_factory(args.arch)

    outcomes = []
    discarded: list[Any] = []
    for config in enumerate_configurations(spec):
        # Per-config kill direction comes from the registry (encoder
        # tasks may report higher-is-better metrics; decoder LM uses
        # lower-is-better loss). Don't tie this to ``args.arch`` —
        # future tasks may break that assumption (e.g. ROUGE on
        # decoder summarization).
        task_spec = get_task_spec(config.task)
        outcome = run_one_configuration(
            config=config,
            injector_factory=_lazy_injector_factory,
            fine_tune=fine_tune,
            feature_builder=default_feature_builder,
            higher_is_better=task_spec.higher_is_better,
            alpha=args.alpha,
            verifier_factory=verifier_factory,
            seeds=seeds,
            output_dir=Path(args.status_dir) if args.status_dir else None,
        )
        if outcome.ok:
            outcomes.append(outcome)
            assert outcome.mutant is not None  # narrowed by status==ok
            print(f"[defaultpp-benchmark] {config.config_id()} "
                  f"killed={outcome.mutant.killed} "
                  f"p={outcome.mutant.p_value:.5g}")
        else:
            discarded.append(outcome)
            print(f"[defaultpp-benchmark] DISCARDED {config.config_id()} "
                  f"({outcome.status.value}): {outcome.discard_reason}")

    # ── Correct class: label-preserving clean variants ───────────────
    # For each (model, task), generate clean variants of the base model
    # and keep the ones that stay statistically indistinguishable from
    # the base model under the same kill test. In the paper, the variant
    # count per base model equals its killed-mutant count k; here it is
    # the --clean-variants value.
    correct_samples: list[Any] = []
    if args.clean_variants > 0:
        clean_fine_tune = make_clean_fine_tune(fine_tune)
        base_hp = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
        for model_name in _parse_csv(args.models):
            for task in tasks:
                task_spec = get_task_spec(task)
                variants = generate_clean_variants(
                    model_name, task, args.clean_variants, base_seed=seeds[0])
                for variant in variants:
                    sample = run_one_clean_variant(
                        variant, clean_fine_tune, default_feature_builder,
                        higher_is_better=task_spec.higher_is_better,
                        alpha=args.alpha, seeds=seeds, base_hyperparams=base_hp,
                    )
                    if sample.retained:
                        correct_samples.append(sample)
                        print(f"[defaultpp-benchmark] CORRECT {variant.config_id()} "
                              f"p={sample.p_value:.5g}")
                    else:
                        print(f"[defaultpp-benchmark] DISCARDED {variant.config_id()} "
                              f"(clean): {sample.rejected_reason}")

    if outcomes or correct_samples:
        feature_key_sets = [o.mutant.feature_vector.keys() for o in outcomes]
        feature_key_sets += [
            s.feature_vector.keys() for s in correct_samples if s.feature_vector
        ]
        feature_columns = sorted(set().union(*feature_key_sets)) if feature_key_sets else []
        writer = DatasetWriter(output, fixed_columns=feature_columns)
        for outcome in outcomes:
            writer.append(outcome.mutant)
        for sample in correct_samples:
            writer.append_correct_sample(sample)

    if discarded:
        discard_log = output.with_name(f"{output.stem}.discarded.jsonl")
        _write_discard_log(discard_log, discarded)
        print(f"[defaultpp-benchmark] wrote discard log to {discard_log}")

    n_rows = len(outcomes) + len(correct_samples)
    total = len(outcomes) + len(discarded)
    print(f"[defaultpp-benchmark] wrote {n_rows} row(s) to {output} "
          f"({len(outcomes)} faulty, {len(correct_samples)} correct); "
          f"{len(discarded)} faulty config(s) discarded")
    if discarded:
        _print_discard_summary(discarded)
    return 0


def _build_verifier_factory(arch: str):
    """Return a verifier_factory closure for the runner.

    On every call it loads a *throwaway* fresh copy of the model
    (so the verifier never mutates the model used for fine-tuning),
    constructs the configured injector, and dispatches to the right
    :class:`StructuralVerifier` method based on the injector type.
    Returns ``None`` if construction fails so the runner records the
    failure as a verifier discard rather than crashing.
    """
    from ..deform.injection import DynamicFault, StaticFault
    from ..deform.validation import StructuralVerifier, VerificationResult
    from ..deform import get_expected_modules, get_expected_parameter_names

    verifier = StructuralVerifier()

    def _factory(config):
        try:
            model = _construct_throwaway_model(arch, config.model)
        except Exception as exc:
            return VerificationResult(
                ok=False,
                message=(f"could not load throwaway model {config.model!r} "
                         f"for verification: {type(exc).__name__}: {exc}"),
            )

        try:
            bind = _lazy_injector_factory(config)
            injector = bind(model)
        except Exception as exc:
            return VerificationResult(
                ok=False,
                message=(f"injector construction failed: "
                         f"{type(exc).__name__}: {exc}"),
            )

        try:
            if isinstance(injector, StaticFault):
                expected = get_expected_parameter_names(model, injector)
                return verifier.verify_static(model, injector, expected)
            if isinstance(injector, DynamicFault):
                expected_modules = get_expected_modules(model, injector)
                return verifier.verify_dynamic(model, injector, expected_modules)
            # Attribute-style faults exercise their own context manager
            # round-trip; success is an empty enter/exit cycle.
            with injector:
                pass
            return VerificationResult(ok=True)
        except Exception as exc:
            return VerificationResult(
                ok=False,
                message=(f"verifier raised {type(exc).__name__}: {exc}"),
            )

    return _factory


def _construct_throwaway_model(arch: str, model_name: str) -> nn.Module:
    if arch == "encoder":
        from transformers import AutoModelForSequenceClassification
        return _from_pretrained_eager_attention(
            AutoModelForSequenceClassification, model_name, num_labels=2,
        )
    from transformers import AutoModelForCausalLM
    return _from_pretrained_eager_attention(AutoModelForCausalLM, model_name)


def _write_discard_log(path: Path, discarded: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with path.open("w") as f:
        for outcome in discarded:
            f.write(json.dumps(outcome.discard_record(), default=str) + "\n")


def _print_discard_summary(discarded: list) -> None:
    counts: dict[str, int] = {}
    for outcome in discarded:
        counts[outcome.status.value] = counts.get(outcome.status.value, 0) + 1
    parts = ", ".join(f"{status}={n}" for status, n in sorted(counts.items()))
    print(f"[defaultpp-benchmark] discard summary: {parts}")


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


_GLUE_NUM_LABELS = {
    "sst2": 2, "qnli": 2, "rte": 2, "mrpc": 2, "qqp": 2, "cola": 2,
    "mnli": 3,
    "stsb": 1,  # regression head
}


def _load_encoder_dataset(task: str,
                          tokenizer: Any,
                          max_train: int,
                          max_eval: int,
                          *,
                          synthetic_fallback: bool):
    num_labels = _GLUE_NUM_LABELS.get(task, 2)
    is_regression = task == "stsb"

    if task in _GLUE_TASKS:
        try:
            from datasets import load_dataset

            raw = load_dataset("glue", task)
            text_cols = _glue_text_columns(task)

            def tokenize(batch):
                if len(text_cols) == 1:
                    out = tokenizer(batch[text_cols[0]], padding="max_length",
                                    truncation=True, max_length=64)
                else:
                    out = tokenizer(batch[text_cols[0]], batch[text_cols[1]],
                                    padding="max_length", truncation=True, max_length=64)
                # Regression heads need float labels for HF's MSE loss.
                if is_regression and "label" in batch:
                    out["label"] = [float(v) for v in batch["label"]]
                return out

            train = raw["train"].select(range(min(max_train, len(raw["train"]))))
            eval_name = "validation_matched" if task == "mnli" else "validation"
            eval_ds = raw[eval_name].select(range(min(max_eval, len(raw[eval_name]))))
            train = train.map(tokenize, batched=True).rename_column("label", "labels")
            eval_ds = eval_ds.map(tokenize, batched=True).rename_column("label", "labels")
            cols = ["input_ids", "attention_mask", "labels"]
            return (train.with_format("torch", columns=cols),
                    eval_ds.with_format("torch", columns=cols),
                    num_labels)
        except Exception:
            if not synthetic_fallback:
                raise

    if not synthetic_fallback:
        raise ValueError(f"No built-in loader for encoder task {task!r}")
    return _synthetic_encoder_dataset(
        tokenizer, max_train, max_eval, num_labels=num_labels,
        regression=is_regression,
    )


def _load_decoder_dataset(task: str,
                          tokenizer: Any,
                          max_train: int,
                          max_eval: int,
                          *,
                          synthetic_fallback: bool):
    try:
        from datasets import load_dataset

        if task in ("wikitext2", "wikitext"):
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


def _synthetic_encoder_dataset(tokenizer: Any,
                               max_train: int,
                               max_eval: int,
                               *,
                               num_labels: int = 2,
                               regression: bool = False):
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
        if regression:
            # Map binary labels to bounded float scores in [1.0, 5.0]
            # so the regression head has a continuous target to fit.
            enc["labels"] = torch.tensor(
                [float(r[1]) * 4.0 + 1.0 for r in rows], dtype=torch.float32,
            )
        else:
            label_values = [r[1] % num_labels for r in rows]
            enc["labels"] = torch.tensor(label_values, dtype=torch.long)
        return _TorchDictDataset(enc)

    return build(max_train), build(max_eval), num_labels


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
        "cola": ("sentence",),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "stsb": ("sentence1", "sentence2"),
    }[task]


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


def _validate_tasks(tasks: tuple[str, ...], *, arch: str) -> None:
    """Reject unsupported task ids and arch/task mismatches up front.

    The registry is the single source of truth: every ``--tasks`` entry
    must appear in ``TASK_METRICS`` and its registered ``arch`` must
    match ``--arch``. Failing here surfaces a useful error before we
    spend time loading models.
    """
    if not tasks:
        raise SystemExit("--tasks must list at least one task id")
    unknown = []
    arch_mismatch = []
    for task in tasks:
        try:
            spec = get_task_spec(task)
        except KeyError:
            unknown.append(task)
            continue
        if spec.arch != arch:
            arch_mismatch.append((task, spec.arch))
    if unknown:
        raise SystemExit(
            f"Unknown task(s): {', '.join(unknown)}. "
            f"Supported tasks: {', '.join(supported_tasks())}"
        )
    if arch_mismatch:
        details = ", ".join(f"{t}(arch={a})" for t, a in arch_mismatch)
        raise SystemExit(
            f"--arch={arch!r} is incompatible with task(s) {details}. "
            f"Supported {arch} tasks: {', '.join(supported_tasks(arch))}"
        )


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
