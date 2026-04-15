"""
Data loading utilities for decoder models.

Handles WikiText-2, PTB, Lambada, PersonaChat (LM), HellaSwag, PIQA, ARC (MC tasks).
Supports models: distilgpt2, gpt2, gpt-neo-125m, opt-125m.
"""

import os
from typing import Any, Dict, Optional, Tuple, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.constants import (
    DEFAULT_DECODER_BATCH_SIZE,
    DEFAULT_DECODER_MAX_LENGTH,
    DEFAULT_SINGLE_SEED,
)


# Task type to text field mappings
DECODER_TEXT_FIELDS = {
    "wikitext-2": ("text", None),
    "ptb_text_only": ("sentence", None),
    "lambada": ("text", None),
    "personachat": ("personality", "utterances"),
    "hellaswag": ("ctx", "endings"),
    "piqa": ("goal", "sol"),
    "arc": ("question", "choices"),
}

_DECODER_LOADER_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Dict[str, Any]]] = {}


class LanguageModelingDataLoader:
    """
    Data loader for language modeling datasets.

    Handles WikiText-2, PTB, Lambada, and PersonaChat for causal language modeling.
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        subset: Optional[str] = "wikitext-2-v1",
        model_name: str = "gpt2",
        max_length: int = DEFAULT_DECODER_MAX_LENGTH,
        train_split: str = "train",
        eval_split: str = "validation",
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_DECODER_BATCH_SIZE,
        num_workers: int = 0,
        seed: int = DEFAULT_SINGLE_SEED,
        use_dataset_cache: bool = True,
    ):
        """
        Initialize language modeling data loader.

        Args:
            dataset_name: Dataset name (wikitext, ptb_text_only, lambada, personachat)
            subset: Dataset subset (e.g., wikitext-2-v1 for wikitext)
            model_name: Name of the model (for tokenizer)
            max_length: Maximum sequence length
            train_split: Dataset split to use for training
            eval_split: Dataset split to use for evaluation
            cache_dir: Directory to cache dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            seed: Random seed for reproducibility
            use_dataset_cache: Whether to use cached tokenized datasets
        """
        self.dataset_name = dataset_name
        self.subset = subset
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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set padding_side to 'left' for decoder-only models to avoid right-padding warnings
        self.tokenizer.padding_side = 'left'

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def load_data(self) -> Tuple:
        """
        Load and tokenize language modeling dataset.

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load dataset (cache_dir handled by HF_DATASETS_CACHE env var)
        if self.dataset_name == "wikitext":
            self.dataset = load_dataset("wikitext", self.subset, cache_dir=self.cache_dir)
            text_column = "text"
        elif self.dataset_name == "ptb_text_only":
            self.dataset = load_dataset("ptb_text_only", cache_dir=self.cache_dir)
            text_column = "sentence"
        elif self.dataset_name == "lambada":
            subset = self.subset or "plain_text"
            self.dataset = load_dataset("lambada", subset, cache_dir=self.cache_dir)
            text_column = "text"
        elif self.dataset_name == "conv_ai_2":
            subset = self.subset or "original"
            self.dataset = load_dataset("conv_ai_2", subset, cache_dir=self.cache_dir)
            text_column = None
        elif self.dataset_name == "openwebtext":
            # Load slices directly to avoid tokenizing 8M+ examples
            from datasets import DatasetDict
            subset = self.subset or "plain_text"
            train_data = load_dataset("openwebtext", subset, split=self.train_split, cache_dir=self.cache_dir)
            eval_data = load_dataset("openwebtext", subset, split=self.eval_split, cache_dir=self.cache_dir)
            self.dataset = DatasetDict({"train": train_data, "validation": eval_data})
            self.train_split = "train"
            self.eval_split = "validation"
            text_column = "text"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        def extract_personachat_texts(examples):
            texts = []
            personalities = examples.get("personality") or []
            utterances_list = examples.get("utterances") or []
            for idx, utterances in enumerate(utterances_list):
                history_text = ""
                if utterances:
                    last = utterances[-1]
                    history = last.get("history") or []
                    if history:
                        history_text = " ".join(history)
                persona = ""
                if idx < len(personalities):
                    persona = " ".join(personalities[idx])
                combined = f"{persona} {history_text}".strip()
                if not combined and utterances:
                    candidates = utterances[-1].get("candidates") or []
                    if candidates:
                        combined = candidates[0]
                texts.append(combined)
            return texts

        # Tokenize function for causal LM
        def tokenize_function(examples):
            # Replace empty texts with space (batch size must be preserved for map)
            if text_column:
                texts = [(text if text and len(text.strip()) > 0 else " ") for text in examples[text_column]]
            else:
                texts = [(text if text and len(text.strip()) > 0 else " ") for text in extract_personachat_texts(examples)]

            # Don't use return_tensors="pt" with datasets.map - it expects lists/arrays
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            # For causal LM, labels are the same as input_ids
            # Replace padding tokens with -100 so PyTorch's CrossEntropyLoss ignores them
            # This is the standard HuggingFace/PyTorch convention used by all causal LM models
            # (GPT-2, GPT-Neo, Pythia, LLaMA, etc.) and is equivalent to using
            # DataCollatorForLanguageModeling with mlm=False
            labels = []
            for input_ids in tokenized["input_ids"]:
                # Replace padding token IDs with -100
                label_ids = [(-100 if token_id == self.tokenizer.pad_token_id else token_id)
                            for token_id in input_ids]
                labels.append(label_ids)

            tokenized["labels"] = labels
            return tokenized

        remove_columns = [text_column] if text_column else self.dataset[self.train_split].column_names

        # Tokenize datasets
        # Use cached tokenized data unless explicitly disabled.
        tokenized_datasets = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=self.use_dataset_cache
        )

        # Filter out empty examples
        tokenized_datasets = tokenized_datasets.filter(
            lambda example: len(example["input_ids"]) > 0
        )

        # Set format for PyTorch
        tokenized_datasets.set_format("torch")

        # Split into train and validation
        available_splits = set(tokenized_datasets.keys())
        if self.train_split not in available_splits:
            raise ValueError(
                f"Train split '{self.train_split}' not found in dataset. "
                f"Available splits: {sorted(available_splits)}"
            )
        if self.eval_split not in available_splits:
            raise ValueError(
                f"Eval split '{self.eval_split}' not found in dataset. "
                f"Available splits: {sorted(available_splits)}"
            )

        self.train_dataset = tokenized_datasets[self.train_split]
        self.val_dataset = tokenized_datasets[self.eval_split]

        return self.train_dataset, self.val_dataset

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training DataLoader."""
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
            collate_fn=self._collate_lm_batch
        )

    def get_val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self.val_dataset is None:
            self.load_data()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_lm_batch
        )

    def _collate_lm_batch(self, batch):
        """Collate LM batches while enforcing left-padding and skipping None entries."""
        batch = [example for example in batch if example is not None]
        if not batch:
            raise ValueError("Empty batch after filtering None entries.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for example in batch:
            input_ids = example["input_ids"]
            attention_mask = example.get("attention_mask")
            labels = example.get("labels")

            if not torch.is_tensor(input_ids):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if attention_mask is None:
                attention_mask = (input_ids != pad_token_id).long()
            elif not torch.is_tensor(attention_mask):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            if labels is not None and not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long)
            if labels is None:
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

            # If right padding is detected, shift to left padding.
            if attention_mask.numel() > 0 and attention_mask[-1].item() == 0:
                non_pad = int(attention_mask.sum().item())
                pad_count = int(input_ids.numel() - non_pad)
                if pad_count > 0 and non_pad > 0:
                    token_ids = input_ids[:non_pad]
                    input_ids = torch.cat(
                        [input_ids.new_full((pad_count,), pad_token_id), token_ids],
                        dim=0
                    )
                    attention_mask = torch.cat(
                        [attention_mask.new_zeros(pad_count), attention_mask.new_ones(non_pad)],
                        dim=0
                    )
                    if labels is not None:
                        token_labels = labels[:non_pad]
                        labels = torch.cat(
                            [labels.new_full((pad_count,), -100), token_labels],
                            dim=0
                        )

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        batch_dict = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
        }
        batch_dict["labels"] = torch.stack(labels_list, dim=0)
        return batch_dict


class MultipleChoiceDataLoader:
    """
    Data loader for multiple choice tasks.

        Handles HellaSwag, PIQA, and ARC.
    """

    def __init__(
        self,
        task_name: str,
        subset: Optional[str] = None,
        model_name: str = "gpt2",
        max_length: int = DEFAULT_DECODER_MAX_LENGTH,
        train_split: str = "train",
        eval_split: str = "validation",
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_DECODER_BATCH_SIZE,
        num_workers: int = 0,
        seed: int = DEFAULT_SINGLE_SEED,
    ):
        """
        Initialize multiple choice data loader.

        Args:
            task_name: Task name (hellaswag, piqa, arc)
            subset: Task subset if applicable
            model_name: Name of the model (for tokenizer)
            max_length: Maximum sequence length
            train_split: Dataset split to use for training
            eval_split: Dataset split to use for evaluation
            cache_dir: Directory to cache dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of workers
            seed: Random seed
        """
        self.task_name = task_name.lower()
        self.subset = subset
        self.model_name = model_name
        self.max_length = max_length
        self.train_split = train_split
        self.eval_split = eval_split
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set padding_side to 'left' for decoder-only models to avoid right-padding warnings
        self.tokenizer.padding_side = 'left'

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def _prepare_hellaswag(self, examples):
        """Prepare HellaSwag examples."""
        contexts = examples["ctx"]
        endings_list = examples["endings"]
        labels = examples.get("label", None)

        prepared = []
        for i, (ctx, endings) in enumerate(zip(contexts, endings_list)):
            # Tokenize context + each ending
            choices = []
            for ending in endings:
                full_text = ctx + " " + ending
                tokenized = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                choices.append({
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0)
                })

            item = {"choices": choices}
            if labels is not None:
                item["label"] = int(labels[i])
            prepared.append(item)

        return prepared

    def _prepare_piqa(self, examples):
        """Prepare PIQA examples."""
        goals = examples["goal"]
        sol1_list = examples["sol1"]
        sol2_list = examples["sol2"]
        labels = examples.get("label", None)

        prepared = []
        for i, (goal, sol1, sol2) in enumerate(zip(goals, sol1_list, sol2_list)):
            choices = []
            for sol in [sol1, sol2]:
                full_text = goal + " " + sol
                tokenized = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                choices.append({
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0)
                })

            item = {"choices": choices}
            if labels is not None:
                item["label"] = int(labels[i])
            prepared.append(item)

        return prepared

    def _prepare_arc(self, examples):
        """Prepare ARC examples."""
        questions = examples["question"]
        choices_list = examples["choices"]
        labels = examples.get("answerKey", None)

        prepared = []
        for i, (question, choices_dict) in enumerate(zip(questions, choices_list)):
            choice_texts = choices_dict["text"]
            choice_labels = choices_dict["label"]

            choices = []
            for choice_text in choice_texts:
                full_text = question + " " + choice_text
                tokenized = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                choices.append({
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0)
                })

            item = {"choices": choices}
            if labels is not None:
                # Convert A, B, C, D to 0, 1, 2, 3
                label_str = labels[i]
                if label_str in choice_labels:
                    item["label"] = choice_labels.index(label_str)
                else:
                    item["label"] = 0
            prepared.append(item)

        return prepared

    def load_data(self) -> Tuple:
        """
        Load and tokenize multiple choice dataset.

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load dataset (cache_dir handled by HF_DATASETS_CACHE env var)
        if self.task_name == "hellaswag":
            self.dataset = load_dataset("hellaswag", cache_dir=self.cache_dir)
        elif self.task_name == "piqa":
            self.dataset = load_dataset("piqa", cache_dir=self.cache_dir)
        elif self.task_name == "arc":
            subset = self.subset or "ARC-Easy"
            self.dataset = load_dataset("ai2_arc", subset, cache_dir=self.cache_dir)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        # Prepare examples based on task type
        if self.task_name == "hellaswag":
            prepare_fn = self._prepare_hellaswag
        elif self.task_name == "piqa":
            prepare_fn = self._prepare_piqa
        elif self.task_name == "arc":
            prepare_fn = self._prepare_arc

        # Process datasets
        available_splits = set(self.dataset.keys())
        if self.train_split not in available_splits:
            raise ValueError(
                f"Train split '{self.train_split}' not found in dataset. "
                f"Available splits: {sorted(available_splits)}"
            )
        if self.eval_split not in available_splits:
            raise ValueError(
                f"Eval split '{self.eval_split}' not found in dataset. "
                f"Available splits: {sorted(available_splits)}"
            )

        self.train_dataset = prepare_fn(self.dataset[self.train_split])
        self.val_dataset = prepare_fn(self.dataset[self.eval_split])

        return self.train_dataset, self.val_dataset

    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training DataLoader."""
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
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def get_val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self.val_dataset is None:
            self.load_data()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def _collate_fn(self, batch):
        """Custom collate function for multiple choice data."""
        # Each item in batch has 'choices' (list of dicts) and 'label'
        labels = torch.tensor([item["label"] for item in batch])

        # Stack choices - assuming all items have same number of choices
        num_choices = len(batch[0]["choices"])
        batch_size = len(batch)

        # Initialize tensors
        first_choice = batch[0]["choices"][0]
        seq_len = first_choice["input_ids"].shape[0]

        input_ids = torch.zeros(batch_size, num_choices, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, num_choices, seq_len, dtype=torch.long)

        for i, item in enumerate(batch):
            for j, choice in enumerate(item["choices"]):
                input_ids[i, j] = choice["input_ids"]
                attention_mask[i, j] = choice["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_decoder_task_data(
    task_name: str,
    task_cfg: Dict[str, Any],
    model_name: str,
    batch_size: int = 8,
    num_workers: int = 0,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    use_dataset_cache: bool = True,
) -> Tuple[Optional[DataLoader], DataLoader, Dict[str, Any]]:
    """
    Factory function to load decoder task data.

    Args:
        task_name: Name of the task (wikitext-2, ptb_text_only, lambada, personachat, hellaswag, piqa, arc)
        task_cfg: Task configuration dict from all-decoder.yaml
        model_name: HuggingFace model name (distilgpt2, gpt2, gpt-neo-125m, opt-125m)
        batch_size: Batch size
        num_workers: DataLoader workers
        max_length: Maximum sequence length
        cache_dir: Cache directory
        seed: Random seed
        use_dataset_cache: Whether to use cached tokenized datasets

    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    task_name = task_name.lower()
    task_type = task_cfg.get("task_type", "lm")
    metric = task_cfg.get("metric", "perplexity")

    def _cache_key(
        dataset_name: str,
        subset: Optional[str],
        train_split: str,
        eval_split: str,
    ) -> Tuple[Any, ...]:
        return (
            task_name,
            dataset_name,
            subset,
            model_name,
            batch_size,
            num_workers,
            max_length,
            cache_dir,
            seed,
            use_dataset_cache,
            train_split,
            eval_split,
            task_type,
            metric,
        )

    # Language modeling tasks (WikiText-2, PTB, Lambada, conv_ai_2)
    if task_name == "wikitext-2":
        dataset_name = task_cfg.get("dataset", "wikitext")
        subset = task_cfg.get("subset", "wikitext-2-v1")
        cache_key = _cache_key(
            dataset_name,
            subset,
            task_cfg.get("train_split", "train"),
            task_cfg.get("eval_split", "validation"),
        )
        cached = _DECODER_LOADER_CACHE.get(cache_key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)
        loader = LanguageModelingDataLoader(
            dataset_name=dataset_name,
            subset=subset,
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
        info = {"task_type": task_type, "metric": metric}
        _DECODER_LOADER_CACHE[cache_key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    elif task_name == "ptb_text_only":
        dataset_name = task_cfg.get("dataset", "ptb_text_only")
        cache_key = _cache_key(
            dataset_name,
            None,
            task_cfg.get("train_split", "train"),
            task_cfg.get("eval_split", "validation"),
        )
        cached = _DECODER_LOADER_CACHE.get(cache_key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)
        loader = LanguageModelingDataLoader(
            dataset_name=dataset_name,
            subset=None,
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
        info = {"task_type": task_type, "metric": metric}
        _DECODER_LOADER_CACHE[cache_key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    elif task_name == "lambada":
        dataset_name = task_cfg.get("dataset", "lambada")
        subset = task_cfg.get("subset", "plain_text")
        cache_key = _cache_key(
            dataset_name,
            subset,
            task_cfg.get("train_split", "train"),
            task_cfg.get("eval_split", "validation"),
        )
        cached = _DECODER_LOADER_CACHE.get(cache_key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)
        loader = LanguageModelingDataLoader(
            dataset_name=dataset_name,
            subset=subset,
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
        info = {"task_type": task_type, "metric": metric}
        _DECODER_LOADER_CACHE[cache_key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    elif task_name == "openwebtext":
        dataset_name = task_cfg.get("dataset", "openwebtext")
        subset = task_cfg.get("subset", "plain_text")
        cache_key = _cache_key(
            dataset_name,
            subset,
            task_cfg.get("train_split", "train[:80138]"),
            task_cfg.get("eval_split", "train[80138:96166]"),
        )
        cached = _DECODER_LOADER_CACHE.get(cache_key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)
        loader = LanguageModelingDataLoader(
            dataset_name=dataset_name,
            subset=subset,
            model_name=model_name,
            max_length=max_length,
            train_split=task_cfg.get("train_split", "train[:80138]"),
            eval_split=task_cfg.get("eval_split", "train[80138:96166]"),
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            use_dataset_cache=use_dataset_cache,
        )
        loader.load_data()
        info = {"task_type": task_type, "metric": metric}
        _DECODER_LOADER_CACHE[cache_key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    elif task_name == "conv_ai_2":
        dataset_name = task_cfg.get("dataset", "conv_ai_2")
        subset = task_cfg.get("subset", "original")
        cache_key = _cache_key(
            dataset_name,
            subset,
            task_cfg.get("train_split", "train"),
            task_cfg.get("eval_split", "validation"),
        )
        cached = _DECODER_LOADER_CACHE.get(cache_key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)
        loader = LanguageModelingDataLoader(
            dataset_name=dataset_name,
            subset=subset,
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
        info = {"task_type": task_type, "metric": metric}
        _DECODER_LOADER_CACHE[cache_key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    # Multiple choice tasks (HellaSwag, PIQA, ARC)
    elif task_name in ["hellaswag", "piqa", "arc"]:
        subset = task_cfg.get("subset", None)
        dataset_name = task_cfg.get("dataset", task_name)
        cache_key = _cache_key(
            dataset_name,
            subset,
            task_cfg.get("train_split", "train"),
            task_cfg.get("eval_split", "validation"),
        )
        cached = _DECODER_LOADER_CACHE.get(cache_key)
        if cached is not None:
            loader, info = cached
            return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)
        loader = MultipleChoiceDataLoader(
            task_name=task_name,
            subset=subset,
            model_name=model_name,
            max_length=max_length,
            train_split=task_cfg.get("train_split", "train"),
            eval_split=task_cfg.get("eval_split", "validation"),
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed
        )
        loader.load_data()
        info = {"task_type": task_type, "metric": metric}
        _DECODER_LOADER_CACHE[cache_key] = (loader, info)
        return loader.get_train_dataloader(), loader.get_val_dataloader(), dict(info)

    else:
        raise ValueError(f"Unknown decoder task: {task_name}")
