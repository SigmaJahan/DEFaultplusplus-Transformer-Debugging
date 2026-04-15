"""Data loading utilities for encoder models.

Handles GLUE tasks (SST-2, MNLI, QQP, CoLA, MRPC, RTE, STS-B) and CoNLL-2003 NER.
Supports models: bert-base-uncased, distilbert-base-uncased, roberta-base, electra-small.
"""

import os
from typing import Any, Dict, Optional, Tuple, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.constants import (
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_ENCODER_MAX_LENGTH,
    DEFAULT_SINGLE_SEED,
)

GLUE_TASK_FIELDS = {
    "sst2": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

NER_LABEL_LIST = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
]

_ENCODER_LOADER_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Dict[str, Any]]] = {}


class ClassificationDataLoader:
    def __init__(
        self,
        task_name: str = "sst2",
        model_name: str = "bert-base-uncased",
        max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
        train_split: str = "train",
        eval_split: str = "validation",
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
        num_workers: int = 0,
        seed: int = DEFAULT_SINGLE_SEED,
        use_dataset_cache: bool = True,
    ):
        self.task_name = task_name.lower()
        self.model_name = model_name
        self.max_length = max_length
        self.train_split = train_split
        self.eval_split = eval_split
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_dataset_cache = use_dataset_cache

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

        if self.task_name not in GLUE_TASK_FIELDS:
            raise ValueError(f"Unknown GLUE task: {self.task_name}. Supported: {list(GLUE_TASK_FIELDS.keys())}")

        self.text_field_a, self.text_field_b = GLUE_TASK_FIELDS[self.task_name]
        self.is_regression = self.task_name == "stsb"

    def load_data(self) -> Tuple:
        glue_name = self.task_name if self.task_name != "sst2" else "sst2"
        self.dataset = load_dataset("glue", glue_name, cache_dir=self.cache_dir)

        def tokenize_function(examples):
            if self.text_field_b is not None:
                texts_a = [t if t and len(t.strip()) > 0 else " " for t in examples[self.text_field_a]]
                texts_b = [t if t and len(t.strip()) > 0 else " " for t in examples[self.text_field_b]]
                tokenized = self.tokenizer(
                    texts_a, texts_b,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                texts = [t if t and len(t.strip()) > 0 else " " for t in examples[self.text_field_a]]
                tokenized = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )

            tokenized["labels"] = examples["label"]
            return tokenized

        columns_to_remove = [c for c in self.dataset[self.train_split].column_names
                             if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]

        tokenized_datasets = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            load_from_cache_file=self.use_dataset_cache,
        )

        tokenized_datasets.set_format("torch")

        eval_split = self.eval_split
        if self.task_name == "mnli" and eval_split == "validation":
            eval_split = "validation_matched"

        available = set(tokenized_datasets.keys())
        if self.train_split not in available:
            raise ValueError(f"Train split '{self.train_split}' not found. Available: {sorted(available)}")
        if eval_split not in available:
            raise ValueError(f"Eval split '{eval_split}' not found. Available: {sorted(available)}")

        self.train_dataset = tokenized_datasets[self.train_split]
        self.val_dataset = tokenized_datasets[eval_split]
        return self.train_dataset, self.val_dataset

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.train_dataset is None:
            self.load_data()
        from src.utils.reproducibility import get_generator, seed_worker
        g = get_generator(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker if self.num_workers > 0 else None,
            generator=g,
            pin_memory=True,
        )

    def get_val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.load_data()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class NERDataLoader:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
        train_split: str = "train",
        eval_split: str = "validation",
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
        num_workers: int = 0,
        seed: int = DEFAULT_SINGLE_SEED,
        use_dataset_cache: bool = True,
        label_all_tokens: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.train_split = train_split
        self.eval_split = eval_split
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_dataset_cache = use_dataset_cache
        self.label_all_tokens = label_all_tokens

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.label_list = NER_LABEL_LIST
        self.num_labels = len(self.label_list)
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def load_data(self) -> Tuple:
        self.dataset = load_dataset("conll2003", cache_dir=self.cache_dir)

        def tokenize_and_align(examples):
            tokenized = self.tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            all_labels = []
            for i, ner_tags in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_ids = []
                prev_word_id = None
                for word_id in word_ids:
                    if word_id is None:
                        label_ids.append(-100)
                    elif word_id != prev_word_id:
                        label_ids.append(ner_tags[word_id])
                    else:
                        label_ids.append(ner_tags[word_id] if self.label_all_tokens else -100)
                    prev_word_id = word_id
                all_labels.append(label_ids)
            tokenized["labels"] = all_labels
            return tokenized

        columns_to_remove = self.dataset[self.train_split].column_names

        tokenized_datasets = self.dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=columns_to_remove,
            load_from_cache_file=self.use_dataset_cache,
        )
        tokenized_datasets.set_format("torch")

        available = set(tokenized_datasets.keys())
        if self.train_split not in available:
            raise ValueError(f"Train split '{self.train_split}' not found. Available: {sorted(available)}")
        if self.eval_split not in available:
            raise ValueError(f"Eval split '{self.eval_split}' not found. Available: {sorted(available)}")

        self.train_dataset = tokenized_datasets[self.train_split]
        self.val_dataset = tokenized_datasets[self.eval_split]
        return self.train_dataset, self.val_dataset

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.train_dataset is None:
            self.load_data()
        from src.utils.reproducibility import get_generator, seed_worker
        g = get_generator(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker if self.num_workers > 0 else None,
            generator=g,
            pin_memory=True,
        )

    def get_val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.load_data()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def load_encoder_task_data(
    task_name: str,
    task_cfg: Dict[str, Any],
    model_name: str,
    batch_size: int = 16,
    num_workers: int = 0,
    max_length: int = 128,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    use_dataset_cache: bool = True,
) -> Tuple[Optional[DataLoader], DataLoader, Dict[str, Any]]:
    task_name = task_name.lower()
    task_type = task_cfg.get("task_type", "cls")
    metric = task_cfg.get("metric", "accuracy")

    def _cache_key(ds_name: str) -> Tuple[Any, ...]:
        return (
            task_name, ds_name, model_name, batch_size,
            num_workers, max_length, cache_dir, seed,
            use_dataset_cache, task_type, metric,
        )

    if task_name in ("sst2", "mnli", "qqp", "cola", "mrpc", "rte", "stsb"):
        key = _cache_key(task_name)
        cached = _ENCODER_LOADER_CACHE.get(key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

        eval_split = task_cfg.get("eval_split", "validation")
        loader = ClassificationDataLoader(
            task_name=task_name,
            model_name=model_name,
            max_length=max_length,
            train_split=task_cfg.get("train_split", "train"),
            eval_split=eval_split,
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            use_dataset_cache=use_dataset_cache,
        )
        loader.load_data()

        num_labels = 1 if task_name == "stsb" else len(set(loader.train_dataset["labels"].tolist()))
        info = {"task_type": task_type, "metric": metric, "num_labels": num_labels}
        _ENCODER_LOADER_CACHE[key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    elif task_name == "conll2003":
        key = _cache_key("conll2003")
        cached = _ENCODER_LOADER_CACHE.get(key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

        loader = NERDataLoader(
            model_name=model_name,
            max_length=max_length,
            train_split=task_cfg.get("train_split", "train"),
            eval_split=task_cfg.get("eval_split", "validation"),
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            use_dataset_cache=use_dataset_cache,
        )
        loader.load_data()
        info = {"task_type": "ner", "metric": "f1", "num_labels": loader.num_labels}
        _ENCODER_LOADER_CACHE[key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    else:
        raise ValueError(f"Unknown encoder task: {task_name}")
