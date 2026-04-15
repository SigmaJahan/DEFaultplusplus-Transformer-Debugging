#!/usr/bin/env python3
"""
Cache decoder models and datasets for offline use on compute nodes.

This script downloads and caches all models and datasets specified in the
matrix config file so they're available when running on compute nodes without
internet access.

Usage:
    python scripts/cache-models-datasets.py --matrix-config config/all-decoder.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from datasets import load_dataset
except ImportError:
    print("ERROR: transformers and datasets are required.")
    print("Install with: pip install transformers datasets")
    sys.exit(1)


def cache_model(model_id: str, cache_dir: str):
    """Download and cache a decoder model and its tokenizer."""
    print(f"\n{'='*60}")
    print(f"Caching model: {model_id}")
    print(f"{'='*60}")

    try:
        # Cache tokenizer
        print(f"  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False
        )
        print(f"  ✓ Tokenizer cached")

        # Cache config
        print(f"  → Downloading config...")
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False
        )
        print(f"  ✓ Config cached")

        # Cache model weights (using AutoModelForCausalLM for decoder models)
        print(f"  → Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False,
            low_cpu_mem_usage=True,  # Reduce memory usage during loading
            torch_dtype="auto"       # Use optimal dtype for caching
        )
        print(f"  ✓ Model weights cached")
        print(f"  ✓ Model {model_id} successfully cached")

        # Clean up to free memory
        del model
        del tokenizer
        del config

        # Force garbage collection to free memory immediately
        import gc
        gc.collect()

    except Exception as e:
        print(f"  ✗ ERROR caching model {model_id}: {e}")
        raise


def cache_dataset(dataset_id: str, subset: str, splits: list):
    """Download and cache a dataset."""
    full_name = f"{dataset_id}/{subset}" if subset else dataset_id
    print(f"\n{'='*60}")
    print(f"Caching dataset: {full_name}")
    print(f"{'='*60}")

    try:
        for split in splits:
            print(f"  → Downloading split '{split}'...")
            # Don't use cache_dir parameter - let HF_DATASETS_CACHE env var handle it
            if subset:
                dataset = load_dataset(
                    dataset_id,
                    subset,
                    split=split,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    dataset_id,
                    split=split,
                    trust_remote_code=True
                )
            print(f"  ✓ Split '{split}' cached ({len(dataset)} examples)")

        print(f"  ✓ Dataset {full_name} successfully cached")

    except Exception as e:
        print(f"  ✗ ERROR caching dataset {full_name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Cache decoder models and datasets for offline use")
    parser.add_argument(
        "--matrix-config",
        type=str,
        required=True,
        help="Path to matrix config YAML file"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: $HF_HOME or $SCRATCH/hf-cache)"
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only cache models, skip datasets"
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Only cache datasets, skip models"
    )

    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif "HF_HOME" in os.environ:
        cache_dir = os.environ["HF_HOME"]
    elif "SCRATCH" in os.environ:
        cache_dir = os.path.join(os.environ["SCRATCH"], "hf-cache")
    else:
        cache_dir = str(Path.home() / ".cache" / "huggingface")

    print(f"\n{'#'*60}")
    print(f"# Hugging Face Cache Setup (Decoder Models)")
    print(f"{'#'*60}")
    print(f"Cache directory: {cache_dir}")
    print(f"Matrix config: {args.matrix_config}")

    # Set environment variables for caching
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load matrix config
    config_path = Path(args.matrix_config)
    if not config_path.exists():
        print(f"\nERROR: Config file not found: {args.matrix_config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract models and datasets
    models = config.get("models", {})
    tasks = config.get("tasks", {})

    print(f"\nFound {len(models)} models and {len(tasks)} tasks in config")

    # Cache models
    if not args.datasets_only:
        print(f"\n{'#'*60}")
        print(f"# Caching Decoder Models")
        print(f"{'#'*60}")

        for model_key, model_info in models.items():
            model_id = model_info.get("model_id")
            if not model_id:
                print(f"WARNING: No model_id found for '{model_key}', skipping")
                continue

            try:
                cache_model(model_id, cache_dir)
            except Exception as e:
                print(f"ERROR: Failed to cache model {model_id}: {e}")
                print("Continuing with remaining models...")

    # Cache datasets
    if not args.models_only:
        print(f"\n{'#'*60}")
        print(f"# Caching Datasets")
        print(f"{'#'*60}")

        for task_key, task_info in tasks.items():
            dataset_id = task_info.get("dataset")
            if not dataset_id:
                print(f"WARNING: No dataset found for task '{task_key}', skipping")
                continue

            # Get subset (e.g., wikitext-2-v1 for wikitext)
            subset = task_info.get("subset", None)

            # Get splits to cache
            splits = []
            train_split = task_info.get("train_split", "train")
            eval_split = task_info.get("eval_split", "validation")
            if train_split:
                splits.append(train_split)
            if eval_split and eval_split not in splits:
                splits.append(eval_split)

            try:
                cache_dataset(dataset_id, subset, splits)
            except Exception as e:
                full_name = f"{dataset_id}/{subset}" if subset else dataset_id
                print(f"ERROR: Failed to cache dataset {full_name}: {e}")
                print("Continuing with remaining datasets...")

    print(f"\n{'#'*60}")
    print(f"# Caching Complete!")
    print(f"{'#'*60}")
    print(f"\nAll decoder models and datasets are now cached in: {cache_dir}")
    print(f"Your compute nodes will use these cached files when HF_HOME is set.")
    print(f"\nTo verify, check that the cache directory contains files:")
    print(f"  ls -lah {cache_dir}")


if __name__ == "__main__":
    main()
