"""ABNN Fault Injection Pipeline - HPC Compatible Runner"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Heavy imports (torch, transformers, src.*) are deferred so that lightweight
# operations like --list-configs work instantly on login nodes without loading
# CUDA libraries.
# ---------------------------------------------------------------------------
_heavy_imports_done = False


def _ensure_heavy_imports():
    """Lazy-load torch and all src.* modules on first real use."""
    global _heavy_imports_done
    if _heavy_imports_done:
        return
    _heavy_imports_done = True

    import importlib

    global torch
    torch = importlib.import_module("torch")

    global ModelWrapper
    ModelWrapper = importlib.import_module("src.models.model_wrapper").ModelWrapper

    global Trainer
    Trainer = importlib.import_module("src.pipeline.trainer").Trainer

    global load_decoder_task_data
    load_decoder_task_data = importlib.import_module("src.utils.data_loader").load_decoder_task_data

    global set_seed
    set_seed = importlib.import_module("src.utils.reproducibility").set_seed

    global Logger
    Logger = importlib.import_module("src.utils.logger").Logger

    _storage = importlib.import_module("src.utils.storage")
    global HDF5MetricsStorage, SQLiteDatabase
    HDF5MetricsStorage = _storage.HDF5MetricsStorage
    SQLiteDatabase = _storage.SQLiteDatabase

    _profiler = importlib.import_module("src.utils.profiler")
    global GPUProfiler, ProfileStorage, ProfileMetrics, calculate_parallel_config
    GPUProfiler = _profiler.GPUProfiler
    ProfileStorage = _profiler.ProfileStorage
    ProfileMetrics = _profiler.ProfileMetrics
    calculate_parallel_config = _profiler.calculate_parallel_config


# Constants are lightweight (no external deps) — import eagerly.
from src.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_SEED_LIST,
    BASE_LAYER_DEPTH,
    DEFAULT_MASTER_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_RESULTS_DIR,
)

LOGGER = logging.getLogger("pipeline_runner")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_MODEL_CACHE: Dict[Tuple[str, str, int, bool], Dict[str, Any]] = {}


def _clone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cloned = dict(config)
    if isinstance(config.get("severity_params"), dict):
        cloned["severity_params"] = dict(config["severity_params"])
    return cloned


def _load_master_defaults(config_path: str = DEFAULT_MASTER_CONFIG) -> Dict[str, Any]:
    defaults = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        "max_length": DEFAULT_MAX_LENGTH,
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "warmup_ratio": DEFAULT_WARMUP_RATIO,
        "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
        "logging_steps": DEFAULT_LOGGING_STEPS,
    }
    cfg_path = Path(config_path)

    if yaml is None:
        LOGGER.warning("PyYAML not available, using hardcoded defaults")
        return defaults

    if not cfg_path.exists():
        LOGGER.warning(f"Master config file not found at {cfg_path}, using hardcoded defaults")
        return defaults

    try:
        with open(cfg_path, "r") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as e:
        LOGGER.error(f"Failed to parse YAML config at {cfg_path}: {e}")
        LOGGER.warning("Using hardcoded defaults due to YAML parsing error")
        return defaults
    except Exception as e:
        LOGGER.error(f"Unexpected error loading config from {cfg_path}: {e}")
        LOGGER.warning("Using hardcoded defaults due to loading error")
        return defaults

    training = data.get("training", {})
    dataset = data.get("dataset", {})

    if not training and not dataset:
        LOGGER.warning(f"Master config at {cfg_path} is empty or missing 'training'/'dataset' sections")

    defaults.update({
        "batch_size": training.get("batch_size", defaults["batch_size"]),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", defaults["gradient_accumulation_steps"]),
        "epochs": training.get("epochs", defaults["epochs"]),
        "learning_rate": training.get("learning_rate", defaults["learning_rate"]),
        "weight_decay": training.get("weight_decay", defaults["weight_decay"]),
        "warmup_ratio": training.get("warmup_ratio", defaults["warmup_ratio"]),
        "max_grad_norm": training.get("max_grad_norm", defaults["max_grad_norm"]),
        "logging_steps": training.get("logging_steps", defaults["logging_steps"]),
        "max_length": dataset.get("max_length", defaults["max_length"]),
    })

    LOGGER.info(f"Loaded master config from {cfg_path}")
    return defaults


MASTER_DEFAULTS = _load_master_defaults()


def _log(*args: Any) -> None:
    message = " ".join(str(arg) for arg in args)
    LOGGER.info(message)


def require_min_gpu_memory_gb(min_gb: float) -> None:
    if min_gb <= 0:
        return
    _ensure_heavy_imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this pipeline requires a GPU.")

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    total_gb = props.total_memory / (1024 ** 3)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    free_gb = free_bytes / (1024 ** 3)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        _log(f"CUDA_VISIBLE_DEVICES={visible}")
    _log(f"Detected GPU {device_idx}: {props.name} ({total_gb:.1f} GiB)")
    _log(f"GPU memory: free {free_gb:.1f} GiB / total {total_gb:.1f} GiB")

    if total_gb + 1e-6 < min_gb:
        raise RuntimeError(
            f"GPU memory {total_gb:.1f} GiB is below the required {min_gb:.1f} GiB. "
            "This looks like a MIG slice or a smaller GPU. "
            "Request a full H100 80GB (or lower --min-gpu-mem-gb to override)."
        )
    if free_gb + 1e-6 < min_gb:
        raise RuntimeError(
            f"Free GPU memory {free_gb:.1f} GiB is below the required {min_gb:.1f} GiB. "
            "The GPU appears busy or memory is fragmented. "
            "Check other processes, reduce parallel configs, or retry on a fresh GPU."
        )


def _summarize_config_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fault_id": config.get("fault_id"),
        "fault_name": config.get("fault_name"),
        "category": config.get("category"),
        "layer_idx": config.get("layer_idx"),
        "seed": config.get("seed"),
        "is_baseline": bool(config.get("is_baseline", False)),
        "description": config.get("description"),
    }


def _save_failed_configs(results: List[Dict[str, Any]], results_dir: Path) -> None:
    failed_entries = []
    for entry in results:
        if entry.get("status") != "failed":
            continue
        metadata = entry.get("config_metadata") or {}
        failed_entries.append({
            "config_id": entry.get("config_id"),
            "error": entry.get("error"),
            **metadata,
        })

    if not failed_entries:
        return

    failed_file = results_dir / "failed_configs.json"
    existing_entries: Dict[str, Dict[str, Any]] = {}
    if failed_file.exists():
        try:
            with open(failed_file, "r") as handle:
                data = json.load(handle)
            candidate_list = []
            if isinstance(data, dict):
                candidate_list = data.get("failed") or data.get("configs") or []
            elif isinstance(data, list):
                candidate_list = data
            for item in candidate_list:
                if isinstance(item, dict) and item.get("config_id"):
                    existing_entries[item["config_id"]] = item
        except Exception as exc:
            _log(f"Warning: Could not read existing failed config log at {failed_file}: {exc}")

    for item in failed_entries:
        if item.get("config_id"):
            existing_entries[item["config_id"]] = item

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "failed": list(existing_entries.values()),
    }
    failed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_file, "w") as handle:
        json.dump(payload, handle, indent=2)
    _log(f"Saved {len(failed_entries)} failed configs to {failed_file}")


def _load_failed_config_ids(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Failed config file not found: {file_path}")
    with open(file_path, "r") as handle:
        data = json.load(handle)

    entries = []
    if isinstance(data, dict):
        entries = data.get("failed") or data.get("configs") or data.get("results") or []
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported failed config file format: {file_path}")

    config_ids: List[str] = []
    for item in entries:
        if isinstance(item, str):
            config_ids.append(item)
        elif isinstance(item, dict) and item.get("config_id"):
            config_ids.append(item["config_id"])

    return config_ids


def _remap_layer_index(original_idx: int, target_depth: int, base_depth: int = BASE_LAYER_DEPTH) -> int:
    """Map base-layer index (0..base_depth-1) onto target model depth."""
    if not isinstance(original_idx, int):
        raise ValueError(f"original_idx must be an integer, got {type(original_idx)}")
    if not isinstance(target_depth, int):
        raise ValueError(f"target_depth must be an integer, got {type(target_depth)}")
    if not isinstance(base_depth, int):
        raise ValueError(f"base_depth must be an integer, got {type(base_depth)}")

    if target_depth <= 0:
        raise ValueError(f"target_depth must be positive, got {target_depth}")
    if base_depth <= 0:
        raise ValueError(f"base_depth must be positive, got {base_depth}")
    if original_idx < 0:
        raise ValueError(f"original_idx must be non-negative, got {original_idx}")
    if original_idx >= base_depth:
        LOGGER.warning(
            f"original_idx {original_idx} >= base_depth {base_depth}. "
            f"This may indicate a configuration error. Clamping to valid range."
        )
        original_idx = base_depth - 1

    if base_depth == 1:
        return target_depth // 2
    if target_depth <= base_depth:
        mapped = min(original_idx, target_depth - 1)
    else:
        mapped = round(original_idx * (target_depth - 1) / (base_depth - 1))

    result = int(max(0, min(mapped, target_depth - 1)))
    assert 0 <= result < target_depth, f"Remapping produced invalid index {result} for depth {target_depth}"
    return result


def _is_decoder_model(model_name: str, model_type: Optional[str] = None) -> bool:
    model_lower = model_name.lower()
    type_hint = (model_type or "").lower().strip()
    if type_hint in {"gpt2", "gpt_neo", "opt", "decoder"}:
        return True

    decoder_patterns = {
        "distilgpt2": ["distilgpt2"],
        "gpt2": ["gpt2", "gpt-2"],
        "gpt-neo-125m": ["gpt-neo-125m", "eleutherai/gpt-neo-125m", "gpt-neo", "gpt_neo"],
        "opt-125m": ["opt-125m", "facebook/opt-125m", "opt"],
        "gpt4all": ["gpt4all"],
    }
    for aliases in decoder_patterns.values():
        for pattern in aliases:
            if pattern in model_lower:
                return True
    return False


def _resolve_seed_list(seed_entry: Any) -> List[int]:
    if seed_entry is None:
        return DEFAULT_SEED_LIST
    if isinstance(seed_entry, str):
        key = seed_entry.lower().strip()
        if key == "five":
            return DEFAULT_SEED_LIST
        try:
            return [int(key)]
        except ValueError:
            return DEFAULT_SEED_LIST
    if isinstance(seed_entry, list):
        return [int(item) for item in seed_entry]
    return DEFAULT_SEED_LIST


def _load_matrix_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load matrix configs.")
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    global_cfg = data.get("global", {})
    seeds = _resolve_seed_list(global_cfg.get("seed_list"))
    return {
        "global": global_cfg,
        "models": data.get("models", {}),
        "tasks": data.get("tasks", {}),
        "seeds": seeds,
    }


def _apply_seed_filters(
    configs: List[Dict[str, Any]],
    allowed_seeds: Optional[List[int]],
    append_seed: bool,
) -> List[Dict[str, Any]]:
    if not allowed_seeds:
        return [_clone_config(cfg) for cfg in configs]

    filtered: List[Dict[str, Any]] = []
    for raw in configs:
        base_cfg = _clone_config(raw)
        seed = base_cfg.get("seed")
        if seed is not None:
            if seed in allowed_seeds:
                if append_seed:
                    base_cfg["config_id"] = f"{base_cfg['config_id']}_seed{seed}"
                filtered.append(base_cfg)
            continue
        for seed in allowed_seeds:
            cfg = _clone_config(base_cfg)
            cfg["seed"] = seed
            if append_seed and seed is not None:
                cfg["config_id"] = f"{cfg['config_id']}_seed{seed}"
            filtered.append(cfg)

    return filtered


def _filter_by_config_ids(configs: List[Dict[str, Any]], allowed_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not allowed_ids:
        return configs
    allowed_lookup = {str(cfg_id).strip() for cfg_id in allowed_ids if str(cfg_id).strip()}
    return [cfg for cfg in configs if str(cfg.get("config_id")) in allowed_lookup]


def _prepare_run_list(
    config_content: Dict[str, Any],
    allowed_seeds: Optional[List[int]],
    allowed_ids: Optional[List[str]],
    append_seed: bool,
) -> List[Dict[str, Any]]:
    baseline_configs = _apply_seed_filters(config_content.get("baseline", []), allowed_seeds, append_seed=append_seed)
    fault_configs = _apply_seed_filters(config_content.get("faults", []), allowed_seeds, append_seed=append_seed)
    combined = baseline_configs + fault_configs
    return _filter_by_config_ids(combined, allowed_ids)


def run_single_configuration(
    config: Dict[str, Any],
    config_index: int,
    total_configs: int,
    results_dir: Path,
    experiment: Optional[Dict[str, Any]] = None,
    append_seed_to_id: bool = False,
) -> Dict[str, Any]:
    _ensure_heavy_imports()
    config_start = time.perf_counter()
    prefix_parts: List[str] = []
    if experiment:
        prefix_parts.extend([experiment["model_key"], experiment["task_key"]])
    base_config_id = str(config["config_id"])
    config_id = "__".join(prefix_parts + [base_config_id]) if prefix_parts else base_config_id
    if append_seed_to_id and config.get("seed") is not None:
        seed_suffix = f"_seed{config['seed']}"
        if seed_suffix not in config_id:
            config_id = f"{config_id}{seed_suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 80)
    _log(f"Configuration {config_index}/{total_configs}")
    _log("=" * 80)
    _log(f"Config ID: {config_id}")
    _log(f"Description: {config.get('description', 'N/A')}")
    _log(f"Is Baseline: {config.get('is_baseline')}")
    if not config.get("is_baseline"):
        _log(f"Fault: {config.get('fault_id')} - {config.get('fault_name')}")
        _log(f"Category: {config.get('category')}")
        layer_display = config.get("layer_idx")
        if config.get("mapped_layer_idx") is not None and config.get("mapped_layer_idx") != layer_display:
            layer_display = f"{layer_display} -> {config['mapped_layer_idx']}"
        _log(f"Layer: {layer_display}")
        if config.get("severity_params"):
            _log(f"Severity: {config['severity_params']}")
    _log("=" * 80)

    h5_storage = HDF5MetricsStorage(str(results_dir / "metrics.h5"))
    db_storage = SQLiteDatabase(str(results_dir / "dataset.db"))
    logger = Logger(name=config_id, log_dir=str(log_dir))

    existing = db_storage.get_configuration(config_id)
    if existing and existing.get("status") == "completed":
        _log("⏭️  Configuration already completed, skipping...")
        return {"config_id": config_id, "status": "skipped", "reason": "already_completed"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        _log(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    try:
        seed = config.get("seed")
        if seed is None:
            LOGGER.warning(f"No seed specified for config {config_id}, using default seed 42")
            seed = 42
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid seed value: {seed}. Seed must be a non-negative integer.")
        set_seed(seed)
        _log(f"✓ Random seed set: {seed}")

        metadata = {"description": config.get("description")}
        model_name = None
        dataset_name = None
        if experiment:
            model_name = experiment.get("model_key")
            dataset_name = experiment.get("task_key")
            metadata.update({
                "model_key": model_name,
                "task_key": dataset_name,
            })
        if not config.get("is_baseline") and "mapped_layer_idx" in config:
            metadata["mapped_layer_idx"] = config["mapped_layer_idx"]
            metadata["original_layer_idx"] = config.get("layer_idx")

        db_storage.insert_configuration(
            config_id=config_id,
            seed=seed,
            fault_category=config.get("category", "baseline"),
            fault_subcategory=config.get("fault_name", "clean"),
            is_faulty=not config.get("is_baseline"),
            status="running",
            fault_id=config.get("fault_id"),
            layer_idx=config.get("layer_idx"),
            severity_params=config.get("severity_params"),
            metadata=metadata,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        
        if experiment is None:
            raise RuntimeError(
                "Decoder pipeline requires an experiment/model-task definition (e.g., via --matrix-config); "
                "DistilBERT fallback has been removed."
            )

        task_cfg = experiment["task_cfg"]
        model_cfg = experiment["model_cfg"]
        global_cfg = experiment.get("global_cfg", {})
        task_name = experiment["task_key"]
        model_id = model_cfg.get("model_id") or experiment.get("model_key")
        if not model_id:
            raise ValueError("Model ID is missing; provide model_id in model_cfg or set experiment model_key.")

        is_decoder = _is_decoder_model(model_id, model_cfg.get("model_type"))

        if is_decoder:
            _log(f"\n Loading decoder dataset for task {task_name}...")
            train_dataloader, val_dataloader, test_dataloader = load_decoder_task_data(
                task_name=task_name,
                task_cfg=task_cfg,
                model_name=model_id,
                batch_size=global_cfg.get("batch_size", MASTER_DEFAULTS["batch_size"]),
                num_workers=task_cfg.get(
                    "num_workers",
                    global_cfg.get("num_workers", 0),
                ),
                max_length=task_cfg.get("max_length", MASTER_DEFAULTS["max_length"]),
                cache_dir=task_cfg.get("cache_dir") or global_cfg.get("cache_dir"),
                use_dataset_cache=task_cfg.get(
                    "use_dataset_cache",
                    global_cfg.get("use_dataset_cache", True),
                ),
                seed=seed,
            )
            _log(f"✓ Decoder dataset loaded: {task_name}")

            cache_device_idx = torch.cuda.current_device() if device.type == "cuda" else -1
            cache_key = (
                model_id,
                device.type,
                cache_device_idx,
                bool(global_cfg.get("gradient_checkpointing", False)),
            )
            cached_model = _MODEL_CACHE.get(cache_key)

            if cached_model is None:
                _log(f"\n Loading decoder model {model_id}...")
                import warnings
                warnings.filterwarnings("ignore", message="Some weights of")

                model_wrapper = ModelWrapper(
                    model_name=model_id,
                    device=device,
                    gradient_checkpointing=bool(global_cfg.get("gradient_checkpointing", False)),
                )
                base_state = {
                    name: param.detach().cpu().clone()
                    for name, param in model_wrapper.model.state_dict().items()
                }
                _MODEL_CACHE[cache_key] = {
                    "model_wrapper": model_wrapper,
                    "base_state_dict": base_state,
                }
                _log(f"✓ Decoder model loaded: {model_id}")
            else:
                model_wrapper = cached_model["model_wrapper"]
                base_state = cached_model.get("base_state_dict")
                if base_state is not None:
                    missing, unexpected = model_wrapper.model.load_state_dict(base_state, strict=False)
                    alias_suffixes = (
                        ".attn.q_lin.weight",
                        ".attn.k_lin.weight",
                        ".attn.v_lin.weight",
                        ".attn.out_lin.weight",
                        ".attn.out_lin.bias",
                    )
                    ignore_missing = [k for k in missing if k.endswith(alias_suffixes)]
                    other_missing = [k for k in missing if k not in ignore_missing]
                    if other_missing:
                        _log(f"[WARN] Missing keys after reset: {other_missing[:5]}")
                    if unexpected:
                        _log(f"[WARN] Unexpected keys after reset: {unexpected[:5]}")
                _log(f"✓ Reusing decoder model: {model_id}")
        else:
            raise RuntimeError(
                f"Encoder model detected: {model_id}\n"
                f"This pipeline now supports decoder-only models (GPT-2, GPT-Neo, OPT, etc.).\n"
                f"Encoder support (BERT, RoBERTa, DistilBERT) has been archived to archive/encoder-legacy/.\n"
                f"Please use a decoder model or restore encoder files from archive if needed."
            )

        fault_context = None
        if not config.get("is_baseline"):
            _log(f"\n💉 Injecting fault: {config.get('fault_name')}...")
            category = config.get("category")
            layer_idx = config.get("layer_idx")
            fault_name = config.get("fault_name")
            severity_params = config.get("severity_params") or {}

            mapped_layer_idx = layer_idx
            try:
                target_depth = model_wrapper.get_num_layers()
                base_depth = max(
                    BASE_LAYER_DEPTH,
                    int(config.get("base_depth", 0) or 0),
                    (layer_idx + 1) if isinstance(layer_idx, int) else 0,
                )
                mapped_layer_idx = _remap_layer_index(layer_idx, target_depth, base_depth=base_depth)
                config["mapped_layer_idx"] = mapped_layer_idx
                if mapped_layer_idx != layer_idx:
                    _log(f"Remapped layer {layer_idx} -> {mapped_layer_idx} (target depth={target_depth})")
            except Exception:
                mapped_layer_idx = layer_idx

            if category == "masking":
                from src.faults.masking_faults import create_masking_fault

                fault_context = create_masking_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "qkv":
                from src.faults.qkv_faults import create_qkv_fault

                fault_context = create_qkv_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "score":
                from src.faults.score_faults import create_score_fault

                fault_context = create_score_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "positional":
                from src.faults.positional_faults import create_positional_fault

                fault_context = create_positional_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "kernel":
                from src.faults.kernel_faults import create_kernel_fault

                fault_context = create_kernel_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "variant":
                from src.faults.variant_faults import create_variant_fault

                fault_context = create_variant_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "ffn":
                from src.faults.ffn_faults import create_ffn_fault

                fault_context = create_ffn_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "layernorm":
                from src.faults.layernorm_faults import create_layernorm_fault

                fault_context = create_layernorm_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "residual":
                from src.faults.residual_faults import create_residual_fault

                fault_context = create_residual_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "embedding":
                from src.faults.embedding_faults import create_embedding_fault

                fault_context = create_embedding_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "output":
                from src.faults.output_faults import create_output_fault

                fault_context = create_output_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "decoder_masking":
                from src.faults.decoder_masking_faults import create_decoder_masking_fault

                fault_context = create_decoder_masking_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            elif category == "kv_cache":
                from src.faults.kv_cache_faults import create_kv_cache_fault

                fault_context = create_kv_cache_fault(fault_name, model_wrapper.model, mapped_layer_idx, **severity_params)
            else:
                raise ValueError(f"Unknown fault category: {category}")

            try:
                fault_context.__enter__()
            except Exception as inject_err:
                _log(f"FATAL: Fault injection failed for {config.get('fault_id')}: {inject_err}")
                raise RuntimeError(f"Fault injection failed: {inject_err}") from inject_err
            _log(f"✓ Fault injected: {fault_name} [{category}] layer={mapped_layer_idx} "
                 f"severity={severity_params or 'default'}")

        try:
            training_config = {
                "epochs": int(global_cfg.get("epochs", MASTER_DEFAULTS["epochs"])),
                "batch_size": int(global_cfg.get("batch_size", MASTER_DEFAULTS["batch_size"])),
                "learning_rate": float(model_cfg.get("learning_rate", MASTER_DEFAULTS["learning_rate"])),
                "weight_decay": float(global_cfg.get("weight_decay", MASTER_DEFAULTS["weight_decay"])),
                "warmup_ratio": float(global_cfg.get("warmup_ratio", MASTER_DEFAULTS["warmup_ratio"])),
                "max_grad_norm": float(global_cfg.get("max_grad_norm", MASTER_DEFAULTS["max_grad_norm"])),
                "gradient_accumulation_steps": int(global_cfg.get("gradient_accumulation_steps", MASTER_DEFAULTS["gradient_accumulation_steps"])),
                "logging_steps": int(global_cfg.get("logging_steps", MASTER_DEFAULTS["logging_steps"])),
                "task_info": {
                    "task_name": experiment.get("task_key"),
                    "num_labels": task_cfg.get("num_labels", 2),
                    "regression": task_cfg.get("regression", False),
                    "metric": task_cfg.get("metric"),
                    "secondary_metric": task_cfg.get("secondary_metric"),
                    "task_type": task_cfg.get("task_type", "lm"),
                },
            }

            # Choose appropriate trainer based on model type
            if is_decoder:
                task_type = training_config.get('task_info', {}).get('task_type', 'lm')
                trainer = Trainer(
                    model=model_wrapper,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    device=device,
                    config=training_config,
                    logger=logger,
                    config_id=config_id,
                    task_type=task_type,
                    h5_storage=h5_storage,
                    db_storage=db_storage,
                    run_metadata={
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                    },
                )
            else:
                raise RuntimeError(
                    "Encoder trainer path should not be reached. "
                    "This indicates a bug in model type detection. "
                    "All models should be decoder-only."
                )

            _log(f"\n Starting training ({training_config['epochs']} epochs, "
                 f"batch={training_config['batch_size']}, lr={training_config['learning_rate']})...")
            trainer.train()

            final_metrics = trainer.metric_collector.get_final_metrics()
            _log(f"✓ Training complete. Final features: {len(final_metrics)} metrics collected")
            _log(f"  loss={final_metrics.get('final_loss', '?'):.4f} "
                 f"ppl={final_metrics.get('final_val_perplexity', final_metrics.get('final_eval_perplexity', '?'))}")

            db_storage.update_configuration_results(
                config_id=config_id,
                final_accuracy=final_metrics.get("final_accuracy", 0.0),
                final_loss=final_metrics.get("final_loss", 0.0),
                final_f1_score=final_metrics.get("final_f1_score", 0.0),
                best_accuracy=final_metrics.get("best_accuracy", 0.0),
                best_loss=final_metrics.get("best_loss", float('inf')),
                status="completed",
            )

            config_elapsed = time.perf_counter() - config_start
            gpu_peak = final_metrics.get("runtime_memory_peak_mb_early_mean",
                        final_metrics.get("runtime_memory_alloc_mb_early_mean", 0))
            ram_peak = final_metrics.get("runtime_ram_mb_early_mean", 0)
            _log(f"\n Configuration {config_index}/{total_configs} COMPLETE "
                 f"({config_elapsed:.1f}s, GPU peak: {gpu_peak:.0f}MB, RAM: {ram_peak:.0f}MB)")
            return {
                "config_id": config_id,
                "status": "success",
                "final_val_acc": final_metrics.get("final_accuracy", 0.0),
                "final_val_loss": final_metrics.get("final_loss", 0.0),
                "wall_clock_seconds": round(config_elapsed, 2),
                "gpu_peak_mb": round(gpu_peak, 1),
                "ram_peak_mb": round(ram_peak, 1),
                "config_metadata": _summarize_config_metadata(config),
            }
        finally:
            if fault_context is not None:
                fault_context.__exit__(None, None, None)
                _log("\n✓ Fault restored")
    except Exception as exc:
        error_msg = f"Error in configuration {config_id}: {exc}"
        _log(f"\n{error_msg}")
        traceback.print_exc()

        try:
            db_storage.update_configuration_results(
                config_id=config_id,
                final_accuracy=0.0,
                final_loss=999.0,
                final_f1_score=0.0,
                best_accuracy=0.0,
                best_loss=999.0,
                status="failed",
            )
        except Exception as save_error:
            _log(f" Could not save error state: {save_error}")

        config_elapsed = time.perf_counter() - config_start
        return {
            "config_id": config_id,
            "status": "failed",
            "error": error_msg,
            "wall_clock_seconds": round(config_elapsed, 2),
            "config_metadata": _summarize_config_metadata(config),
        }


def _load_configurations_file(config_file: str) -> Dict[str, Any]:
    path = Path(config_file)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r") as handle:
            return json.load(handle)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to parse YAML config files. Install pyyaml or provide a JSON file."
            ) from exc
        with open(path, "r") as handle:
            return yaml.safe_load(handle)


def _run_config_in_subprocess(args_tuple):
    config, position, total_configs, results_dir_path, experiment, append_seed, gpu_id = args_tuple
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    return run_single_configuration(
        config,
        position,
        total_configs,
        results_dir_path,
        experiment=experiment,
        append_seed_to_id=append_seed,
    )


def _run_parallel(
    run_queue, total_configs, results_dir_path, experiment,
    append_seed, parallel_configs, profiler
):
    import torch.multiprocessing as mp
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    batches = []
    for i in range(0, len(run_queue), parallel_configs):
        batches.append(run_queue[i:i+parallel_configs])

    summary = {"completed": 0, "failed": 0, "skipped": 0, "total": len(run_queue), "results": []}

    mp.set_start_method('spawn', force=True)

    for batch_idx, batch in enumerate(batches, 1):
        _log(f"\n--- Batch {batch_idx}/{len(batches)} ({len(batch)} configs) ---")

        batch_args = []
        for gpu_slot, (position, config) in enumerate(batch):
            batch_args.append((
                config, position, total_configs, results_dir_path,
                experiment, append_seed, gpu_slot % max(1, gpu_count)
            ))

        if profiler:
            profiler.record_config_start()

        with mp.Pool(processes=len(batch)) as pool:
            batch_results = pool.map(_run_config_in_subprocess, batch_args)

        if profiler:
            for _ in batch:
                profiler.record_config_end()

        for result in batch_results:
            summary["results"].append(result)
            status = result.get("status")
            if status == "success":
                summary["completed"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            else:
                summary["failed"] += 1

    return summary


def run_pipeline(
    config_file: str,
    start_index: int = 0,
    max_configs: Optional[int] = None,
    results_dir: str = DEFAULT_RESULTS_DIR,
    allowed_seeds: Optional[List[int]] = None,
    config_ids: Optional[List[str]] = None,
    array_index: Optional[int] = None,
    append_seed: bool = False,
    list_only: bool = False,
    experiment: Optional[Dict[str, Any]] = None,
    profile: bool = False,
    parallel_configs: Optional[int] = None,
) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    results_dir_path = Path(results_dir)
    config_content = _load_configurations_file(config_file)

    filtered_configs = _prepare_run_list(
        config_content,
        allowed_seeds=allowed_seeds,
        allowed_ids=config_ids,
        append_seed=append_seed,
    )
    total_configs = len(filtered_configs)

    _log("=" * 80)
    _log("ABNN Fault Injection Pipeline - HPC Runner")
    _log("=" * 80)
    if allowed_seeds:
        _log(f"Seed filter: {allowed_seeds}")
    _log(f"Loaded {total_configs} configurations after filtering")
    _log(f"Results directory: {results_dir_path.resolve()}")
    _log()

    if total_configs == 0:
        _log("No configurations match the provided filters.")
        return {"completed": 0, "failed": 0, "skipped": 0, "total": 0, "results": []}

    if list_only:
        _log("Filtered configuration roster:")
        for idx, cfg in enumerate(filtered_configs, start=1):
            seed_str = f"seed={cfg.get('seed')}" if cfg.get("seed") is not None else "seed=auto"
            tag = "baseline" if cfg.get("is_baseline") else f"{cfg.get('fault_id')}:{cfg.get('fault_name')}"
            _log(f"{idx:04d} | {cfg['config_id']} | {tag} | {seed_str}")
        return {"completed": 0, "failed": 0, "skipped": 0, "total": total_configs, "results": []}

    _ensure_heavy_imports()

    run_queue: List[Any]
    if array_index is not None:
        if 0 <= array_index < total_configs:
            run_queue = [(array_index + 1, filtered_configs[array_index])]
        elif 1 <= array_index <= total_configs:
            run_queue = [(array_index, filtered_configs[array_index - 1])]
        else:
            raise ValueError(
                f"Array index {array_index} outside range 0-{total_configs - 1} or 1-{total_configs}"
            )
    else:
        if start_index > 0:
            _log(f"Resuming from configuration index {start_index + 1}")
        sliced = filtered_configs[start_index:]
        if max_configs is not None:
            sliced = sliced[:max_configs]
        run_queue = list(enumerate(sliced, start=start_index + 1))

    if not run_queue:
        _log("Nothing to run after applying --start-index/--max-configs.")
        return {"completed": 0, "failed": 0, "skipped": 0, "total": 0, "results": []}

    profiler = None
    if profile:
        profiler = GPUProfiler()
        profiler.start()
        _log("=" * 80)
        _log("GPU PROFILING MODE ENABLED")
        _log("=" * 80)
        _log(f"Running {len(run_queue)} config(s) to profile resources")
        _log(f"Total configs for this model-dataset: {total_configs}")
        _log(f"Profile will extrapolate metrics to all {total_configs} configs")
        _log("=" * 80)

    if parallel_configs and parallel_configs > 1:
        _log(f"Parallel execution mode: {parallel_configs} configs per GPU")
        summary = _run_parallel(
            run_queue, total_configs, results_dir_path, experiment,
            append_seed, parallel_configs, profiler
        )
    else:
        summary = {"completed": 0, "failed": 0, "skipped": 0, "total": len(run_queue), "results": []}
        for position, config in run_queue:
            if profiler:
                profiler.record_config_start()

            result = run_single_configuration(
                config,
                position,
                total_configs,
                results_dir_path,
                experiment=experiment,
                append_seed_to_id=append_seed,
            )

            if profiler:
                profiler.record_config_end()

            summary["results"].append(result)
            status = result.get("status")
            if status == "success":
                summary["completed"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            else:
                summary["failed"] += 1

    pipeline_elapsed = time.perf_counter() - pipeline_start
    pipeline_mins = pipeline_elapsed / 60.0
    pipeline_hrs = pipeline_elapsed / 3600.0

    _log("\nPipeline complete")
    _log("-" * 40)
    _log(f"Total scheduled: {summary['total']}")
    _log(f"Completed: {summary['completed']}")
    _log(f"Skipped: {summary['skipped']}")
    _log(f"Failed: {summary['failed']}")
    if pipeline_hrs >= 1.0:
        _log(f"Wall-clock time: {pipeline_hrs:.2f} hours ({pipeline_elapsed:.1f}s)")
    else:
        _log(f"Wall-clock time: {pipeline_mins:.2f} minutes ({pipeline_elapsed:.1f}s)")
    summary["wall_clock_seconds"] = round(pipeline_elapsed, 2)
    _save_failed_configs(summary["results"], results_dir_path)

    if profiler and experiment:
        profile_metrics = profiler.end()
        model_key = experiment.get('model_key')
        task_key = experiment.get('task_key')
        global_cfg = experiment.get('global_cfg', {})

        if model_key and task_key:
            profile_data = ProfileMetrics(
                model_key=model_key,
                task_key=task_key,
                total_configs=total_configs,
                peak_memory_mb=profile_metrics['peak_memory_mb'],
                avg_time_per_config_sec=profile_metrics['avg_time_per_config_sec'],
                gpu_name=profile_metrics['gpu_name'],
                gpu_memory_total_mb=profile_metrics['gpu_memory_total_mb'],
                timestamp=datetime.now().isoformat(),
                batch_size=global_cfg.get('batch_size', DEFAULT_BATCH_SIZE),
                max_length=global_cfg.get('max_length', DEFAULT_MAX_LENGTH),
                num_epochs=global_cfg.get('epochs', DEFAULT_EPOCHS),
            )

            storage = ProfileStorage()
            storage.save_profile(profile_data)

            _log("\n" + "=" * 80)
            _log("PROFILING RESULTS")
            _log("=" * 80)
            _log(f"Measured from {profile_metrics['total_configs']} config(s):")
            _log(f"  Peak memory: {profile_metrics['peak_memory_mb']:.2f} MB per config")
            _log(f"  Avg time: {profile_metrics['avg_time_per_config_sec']:.2f} sec per config ({profile_metrics['avg_time_per_config_sec']/60:.1f} min)")
            _log(f"  GPU: {profile_metrics['gpu_name']}")
            _log(f"  Total memory: {profile_metrics['gpu_memory_total_mb']:.0f} MB")

            parallel_params = calculate_parallel_config(profile_data)
            _log(f"\nExtrapolated for all {total_configs} configs:")
            _log(f"  Configs per GPU: {parallel_params['configs_per_gpu']}")
            _log(f"  Array job size: {parallel_params['array_size']}")
            _log(f"  SLURM time per job: {parallel_params['slurm_time_str']}")
            _log(f"  Sequential time: {parallel_params['estimated_total_time_sequential_sec']/3600:.1f} hours")
            _log(f"  Parallel time: {parallel_params['estimated_total_time_parallel_sec']/3600:.1f} hours")
            _log(f"  Estimated speedup: {parallel_params['estimated_speedup']:.1f}x")
            _log("")
            _log("Next steps:")
            steps = 1
            get_parallel = Path("scripts/get_parallel_params.py")
            submit_parallel = Path("scripts/submit_parallel.sh")
            if get_parallel.exists():
                _log(f"  {steps}. Check profile: python {get_parallel} {model_key} {task_key}")
                steps += 1
            if submit_parallel.exists():
                _log(f"  {steps}. Submit parallel job: bash {submit_parallel} {model_key} {task_key} 4")
                steps += 1
            if steps == 1:
                _log("  - Helper scripts not found; use the profiling summary above.")
            _log("=" * 80)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABNN fault injection pipeline on HPC")
    parser.add_argument(
        "--config-file",
        default=DEFAULT_PIPELINE_CONFIG,
        help="Path to pipeline configs (YAML or JSON)",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for metrics/logs/database files",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start from configuration index (0-based)")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configurations to run")
    parser.add_argument(
        "--allowed-seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds to keep (others skipped).",
    )
    parser.add_argument(
        "--config-ids",
        type=str,
        default=None,
        help="Comma-separated list of config_ids to run. Matches after seed suffixing.",
    )
    parser.add_argument(
        "--failed-configs-file",
        type=str,
        default=None,
        help="Path to a failed_configs.json file to rerun only failed configs.",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Run only the configuration at this index (0-based or 1-based, useful for Slurm arrays).",
    )
    parser.add_argument(
        "--append-seed-to-config-id",
        action="store_true",
        help="Append `_seed<seed>` to non-baseline config IDs to avoid overwriting multi-seed runs.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print filtered configuration IDs and exit without running them.",
    )
    parser.add_argument(
        "--matrix-config",
        type=str,
        default=None,
        help="Optional matrix config (e.g., config/all-decoder.yaml) to run model/task grid.",
    )
    parser.add_argument(
        "--model-keys",
        type=str,
        default=None,
        help="Comma-separated subset of model keys from matrix config to run.",
    )
    parser.add_argument(
        "--task-keys",
        type=str,
        default=None,
        help="Comma-separated subset of task keys from matrix config to run.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable GPU profiling to collect memory and performance metrics.",
    )
    parser.add_argument(
        "--parallel-configs",
        type=int,
        default=None,
        help="Number of configs to run in parallel on single GPU (requires profiling data).",
    )
    parser.add_argument(
        "--min-gpu-mem-gb",
        type=float,
        default=60.0,
        help="Fail fast if GPU memory is below this threshold (GiB). Set 0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    allowed_seeds = None
    if args.allowed_seeds:
        allowed_seeds = [int(seed) for seed in args.allowed_seeds.replace(",", " ").split()]
    selected_ids = None
    if args.config_ids:
        selected_ids = [item.strip() for item in args.config_ids.split(",") if item.strip()]
    if args.failed_configs_file:
        failed_ids = _load_failed_config_ids(args.failed_configs_file)
        if failed_ids:
            selected_ids = (selected_ids or []) + failed_ids
            seen = set()
            filtered: List[str] = []
            for cid in selected_ids:
                if cid in seen:
                    continue
                seen.add(cid)
                filtered.append(cid)
            selected_ids = filtered
        else:
            _log(f"No config IDs found in {args.failed_configs_file}")
    array_index = args.array_index
    if array_index is None:
        env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task_id:
            array_index = int(env_task_id)

    if not args.list_configs:
        require_min_gpu_memory_gb(args.min_gpu_mem_gb)

    if args.matrix_config:
        matrix = _load_matrix_config(args.matrix_config)
        model_filter = {m.strip() for m in args.model_keys.split(",")} if args.model_keys else None
        task_filter = {t.strip() for t in args.task_keys.split(",")} if args.task_keys else None
        seeds_to_use = allowed_seeds or matrix["seeds"]

        combinations: List[Any] = []
        detection_cache: Dict[str, bool] = {}
        for model_key, model_cfg in matrix["models"].items():
            if model_filter and model_key not in model_filter:
                continue
            model_id = model_cfg.get("model_id", model_key)
            detected_decoder = _is_decoder_model(model_id, model_cfg.get("model_type"))
            if model_key not in detection_cache:
                detection_cache[model_key] = detected_decoder
                if detected_decoder:
                    _log(f"Detected decoder model: {model_key} ({model_id})")
                else:
                    _log(f"Warning: model {model_key} ({model_id}) not recognized as decoder; update model_type/model_id if this is decoder-only.")
            for task_key, task_cfg in matrix["tasks"].items():
                if task_filter and task_key not in task_filter:
                    continue
                combinations.append((model_key, model_cfg, task_key, task_cfg))

        if not combinations:
            _log("No model/task pairs selected from matrix config.")
            exit(0)

        overall: List[Dict[str, Any]] = []
        matrix_start = time.perf_counter()
        for model_key, model_cfg, task_key, task_cfg in combinations:
            combo_dir = Path(args.results_dir) / model_key / task_key
            _log(f"\n=== Running combination: {model_key} × {task_key} ===")
            use_seeds = None if selected_ids else seeds_to_use
            use_append_seed = False if selected_ids else args.append_seed_to_config_id
            summary = run_pipeline(
                config_file=args.config_file,
                start_index=args.start_index,
                max_configs=args.max_configs,
                results_dir=str(combo_dir),
                allowed_seeds=use_seeds,
                config_ids=selected_ids,
                array_index=array_index,
                append_seed=use_append_seed,
                list_only=args.list_configs,
                experiment={
                    "model_key": model_key,
                    "model_cfg": model_cfg,
                    "task_key": task_key,
                    "task_cfg": task_cfg,
                    "global_cfg": matrix.get("global", {}),
                },
                profile=args.profile,
                parallel_configs=args.parallel_configs,
            )
            overall.append({
                "model": model_key,
                "task": task_key,
                "results": summary,
            })

        matrix_elapsed = time.perf_counter() - matrix_start
        matrix_mins = matrix_elapsed / 60.0
        matrix_hrs = matrix_elapsed / 3600.0
        _log("\n=== Matrix run complete ===")
        for entry in overall:
            wc = entry['results'].get('wall_clock_seconds', 0)
            _log(f"{entry['model']} / {entry['task']}: {entry['results']} (wall-clock: {wc:.1f}s)")
        if matrix_hrs >= 1.0:
            _log(f"Total matrix wall-clock: {matrix_hrs:.2f} hours ({matrix_elapsed:.1f}s)")
        else:
            _log(f"Total matrix wall-clock: {matrix_mins:.2f} minutes ({matrix_elapsed:.1f}s)")
        total_failed = sum(entry["results"].get("failed", 0) for entry in overall)
        if total_failed:
            _log(f"Exiting non-zero: {total_failed} configuration(s) failed.")
            sys.exit(2)
    else:
        raise RuntimeError(
            "Decoder pipeline now requires a matrix config (model/task grid). "
            "Pass --matrix-config config/all-decoder.yaml and the desired model/task filters."
        )
