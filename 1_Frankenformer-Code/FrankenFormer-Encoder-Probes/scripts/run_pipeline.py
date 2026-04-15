#!/usr/bin/env python3
"""Main entry point for encoder fault injection pipeline."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import yaml
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_encoder_model
from src.utils.data_loader import load_encoder_task_data
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase
from src.utils.logger import Logger
from src.utils.reproducibility import set_seed, get_device
from src.pipeline.trainer import Trainer
from src.metrics.metric_collector import MetricCollector
from src.faults import ALL_FAULTS
from src.constants import (
    ENCODER_FAULT_CATEGORIES,
    DEFAULT_RESULTS_DIR,
    METRICS_FILENAME,
    DATABASE_FILENAME,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Encoder fault injection pipeline")
    parser.add_argument("--matrix-config", type=str, default="config/matrix_encoder.yaml")
    parser.add_argument("--fault-config", type=str, default="config/pipeline_configs_probes.json")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device index")
    parser.add_argument("--config-index", type=int, default=None, help="Run single config by index")
    parser.add_argument("--list-configs", action="store_true", help="List configs and exit (no CUDA needed)")
    parser.add_argument("--cache-dir", type=str, default="hf-cache")
    return parser.parse_args()


def load_matrix(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_fault_configs(path):
    with open(path) as f:
        return json.load(f)


def create_fault(category, fault_type, params):
    key = fault_type
    if key in ALL_FAULTS:
        return ALL_FAULTS[key]()
    for k, cls in ALL_FAULTS.items():
        if k.lower() == key.lower():
            return cls()
    raise ValueError(f"Unknown fault: category={category}, type={fault_type}")


def run_config(cfg, matrix, device, results_dir, cache_dir):
    config_id = cfg["config_id"]
    model_name = cfg["model_name"]
    task_name = cfg["task_name"]
    seed = cfg.get("seed", 42)
    is_baseline = cfg.get("is_baseline", False)
    fault_category = cfg.get("fault_category")
    fault_type = cfg.get("fault_type")
    fault_params = cfg.get("fault_params", {})
    layer_index = cfg.get("layer_index", 0)

    set_seed(seed)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    logger = Logger(config_id, str(results_path / "logs"))
    logger.section(f"Config: {config_id}")
    logger.info(f"Model: {model_name}, Task: {task_name}, Seed: {seed}")

    task_cfg = matrix.get("tasks", {}).get(task_name, {})
    training_cfg = matrix.get("training", {})
    probe_cfg = matrix.get("probes", {})
    num_labels = task_cfg.get("num_labels", 2)
    task_type = task_cfg.get("task_type", "cls")

    train_dl, val_dl, data_info = load_encoder_task_data(
        task_name=task_name,
        task_cfg=task_cfg,
        model_name=model_name,
        batch_size=training_cfg.get("batch_size", 16),
        max_length=matrix.get("models", [{}])[0].get("max_length", 128),
        cache_dir=cache_dir,
        seed=seed,
    )
    num_labels = data_info.get("num_labels", num_labels)

    model_wrapper = load_encoder_model(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
        cache_dir=cache_dir,
    )

    fault_obj = None
    if not is_baseline and fault_category and fault_type:
        logger.info(f"Injecting fault: {fault_category}/{fault_type} at layer {layer_index}")
        fault_obj = create_fault(fault_category, fault_type, fault_params)
        fault_obj.inject(model_wrapper.model, layer_idx=layer_index or 0, **fault_params)

    h5_storage = HDF5MetricsStorage(str(results_path / METRICS_FILENAME))
    db_storage = SQLiteDatabase(str(results_path / DATABASE_FILENAME))

    db_storage.insert_configuration(
        config_id=config_id,
        seed=seed,
        fault_category=fault_category or "baseline",
        fault_subcategory=fault_type or "none",
        is_faulty=not is_baseline,
        status="running",
        layer_idx=layer_index,
        severity_params=fault_params,
        model_name=model_name,
        dataset_name=task_name,
    )

    collector_config = {
        **probe_cfg,
        "num_hidden_layers": model_wrapper.get_num_layers(),
    }
    metric_collector = MetricCollector(
        device=device,
        collect_per_batch=False,
        collect_per_epoch=True,
        collect_attention=True,
        config=collector_config,
    )

    trainer_config = {
        **training_cfg,
        "num_labels": num_labels,
    }

    trainer = Trainer(
        model=model_wrapper,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        device=device,
        config=trainer_config,
        logger=logger,
        config_id=config_id,
        task_type=task_type,
        h5_storage=h5_storage,
        db_storage=db_storage,
        run_metadata={"model": model_name, "task": task_name, "seed": seed},
        metric_collector=metric_collector,
    )

    try:
        results = trainer.train()
        final = results.get("final", {})
        db_storage.update_configuration_results(
            config_id=config_id,
            final_accuracy=final.get("val_accuracy", 0.0),
            final_loss=final.get("val_loss", 0.0),
            final_f1_score=final.get("val_f1", 0.0),
            best_accuracy=final.get("val_accuracy", 0.0),
            best_loss=final.get("val_loss", 0.0),
            status="completed",
        )
        logger.info(f"Completed: acc={final.get('val_accuracy', 0):.4f}, loss={final.get('val_loss', 0):.4f}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        db_storage.update_configuration_results(
            config_id=config_id,
            final_accuracy=0.0, final_loss=0.0, final_f1_score=0.0,
            best_accuracy=0.0, best_loss=0.0, status="failed",
        )
        raise
    finally:
        if fault_obj is not None:
            fault_obj.restore(model_wrapper.model)
        del model_wrapper, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    matrix = load_matrix(args.matrix_config)
    configs = load_fault_configs(args.fault_config)

    if args.list_configs:
        for i, cfg in enumerate(configs):
            baseline = " [baseline]" if cfg.get("is_baseline") else ""
            print(f"[{i}] {cfg['config_id']} - {cfg['model_name']}/{cfg['task_name']}{baseline}")
        print(f"\nTotal: {len(configs)} configurations")
        return

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = get_device()
    print(f"Device: {device}")

    cache_dir = matrix.get("storage", {}).get("cache_dir", args.cache_dir)

    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(configs):
            print(f"Invalid config index {args.config_index}. Range: 0-{len(configs)-1}")
            sys.exit(1)
        run_config(configs[args.config_index], matrix, device, args.results_dir, cache_dir)
    else:
        for i, cfg in enumerate(configs):
            print(f"\n--- [{i+1}/{len(configs)}] {cfg['config_id']} ---")
            try:
                run_config(cfg, matrix, device, args.results_dir, cache_dir)
            except Exception as e:
                print(f"Config {cfg['config_id']} failed: {e}")
                continue

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
