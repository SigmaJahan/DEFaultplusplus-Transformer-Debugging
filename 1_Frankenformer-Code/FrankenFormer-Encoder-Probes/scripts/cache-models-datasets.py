#!/usr/bin/env python3
"""Pre-cache HuggingFace models and datasets for cluster nodes."""

import os
import sys
from pathlib import Path

CACHE_DIR = os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE", "hf-cache"))
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset

ENCODER_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
]

GLUE_TASKS = ["sst2", "mnli"]


def cache_models():
    print(f"Cache directory: {CACHE_DIR}\n")
    for model_name in ENCODER_MODELS:
        print(f"Caching model: {model_name}")
        try:
            AutoConfig.from_pretrained(model_name, cache_dir=CACHE_DIR)
            AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
            AutoModelForSequenceClassification.from_pretrained(
                model_name, cache_dir=CACHE_DIR, num_labels=2,
            )
            print(f"  OK: {model_name}")
        except Exception as e:
            print(f"  FAILED: {model_name}: {e}")


def cache_datasets():
    ds_cache = os.path.join(CACHE_DIR, "datasets")
    for task in GLUE_TASKS:
        print(f"Caching dataset: glue/{task}")
        try:
            load_dataset("glue", task, cache_dir=ds_cache)
            print(f"  OK: {task}")
        except Exception as e:
            print(f"  FAILED: {task}: {e}")


def main():
    print("Pre-caching HuggingFace models and datasets\n")
    cache_models()
    print()
    cache_datasets()
    print("\nDone.")


if __name__ == "__main__":
    main()
