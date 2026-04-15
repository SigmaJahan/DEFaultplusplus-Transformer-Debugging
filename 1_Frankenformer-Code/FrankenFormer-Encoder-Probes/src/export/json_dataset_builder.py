"""JSON dataset exporter for ABNN Encoder Fault Injection runs.

Reads configuration metadata from SQLite/HDF5, hydrates it with master
configuration defaults, and emits per-configuration JSON records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from src.utils.config_manager import ConfigManager
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase


class JSONDatasetBuilder:
    """Builds the final JSON dataset matching the requested schema."""

    def __init__(
        self,
        master_config_path: str,
        h5_path: str,
        db_path: str,
        pipeline_config_path: Optional[str] = None,
    ):
        self.config_manager = ConfigManager(master_config_path)
        self.master_config = self.config_manager.config
        self.h5_storage = HDF5MetricsStorage(h5_path)
        self.db = SQLiteDatabase(db_path)
        self.pipeline_configs = self._load_pipeline_configs(pipeline_config_path)
        self.config_lookup = self._build_config_lookup(self.pipeline_configs)

    def build(self, output_path: str, indent: int = 2) -> List[Dict[str, Any]]:
        config_ids = sorted(self.h5_storage.list_configurations())
        records: List[Dict[str, Any]] = []
        for config_id in config_ids:
            record = self._build_record(config_id)
            if record:
                records.append(record)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(records, f, indent=indent)
        return records

    def _build_record(self, config_id: str) -> Optional[Dict[str, Any]]:
        metrics_blob = self.h5_storage.load_configuration_metrics(config_id)
        if metrics_blob is None:
            return None
        config_row = self.db.get_configuration(config_id)
        if config_row is None:
            return None
        kill_row = self.db.get_kill_result(config_id)
        metadata = metrics_blob.get("metadata") or {}
        run_metadata = metadata.get("run_metadata") or {}
        run_config = run_metadata.get("config") or self.config_lookup.get(str(config_id), {})

        config_section = self._build_config_section(config_row, metadata, run_config)
        labels = self._build_labels_section(config_row, kill_row)
        epochs = self._build_epoch_rows(
            metrics_blob.get("epoch_metrics") or [],
            metrics_blob.get("validation_metrics") or [],
        )
        return {
            "config_id": str(config_id),
            "config": config_section,
            "labels": labels,
            "epochs": epochs,
        }

    def _build_config_section(self, config_row, metadata, run_config) -> Dict[str, Any]:
        dataset_cfg = self.master_config.get("dataset", {})
        training_cfg = metadata.get("training") or self.master_config.get("training", {})
        model_info = metadata.get("model_info") or {}

        model_section = self._build_model_section(model_info, training_cfg)
        training_section = self._build_training_section(training_cfg, metadata.get("run_metadata"))
        fault_section = self._build_fault_section(config_row, run_config, model_section)

        return {
            "task": dataset_cfg.get("task"),
            "dataset": dataset_cfg.get("name"),
            "model": model_section,
            "training": training_section,
            "fault": fault_section,
        }

    @staticmethod
    def _build_model_section(model_info, training_cfg) -> Dict[str, Any]:
        name = model_info.get("model_name") or training_cfg.get("model_name") or "bert-base-uncased"
        num_layers = int(model_info.get("num_layers", training_cfg.get("num_hidden_layers", 12)))
        mapping = {f"L{idx}": f"encoder.layer.{idx}" for idx in range(num_layers)}
        return {
            "name": name,
            "num_layers": num_layers,
            "hidden_size": model_info.get("hidden_size", 768),
            "num_heads": model_info.get("num_attention_heads", 12),
            "vocab_size": model_info.get("vocab_size", 30522),
            "max_position_embeddings": model_info.get("max_position_embeddings", 512),
            "layer_mapping": mapping,
        }

    def _build_training_section(self, training_cfg, run_metadata) -> Dict[str, Any]:
        section = {
            "seed": (run_metadata or {}).get("seed"),
            "num_epochs": training_cfg.get("epochs"),
            "batch_size": training_cfg.get("batch_size"),
            "optimizer": "AdamW",
            "learning_rate": training_cfg.get("learning_rate"),
            "weight_decay": training_cfg.get("weight_decay"),
            "warmup_ratio": training_cfg.get("warmup_ratio"),
            "max_grad_norm": training_cfg.get("max_grad_norm"),
            "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
            "fp16": training_cfg.get("fp16"),
            "gradient_checkpointing": training_cfg.get("gradient_checkpointing"),
        }
        scheduler_cfg = self.master_config.get("scheduler", {})
        if scheduler_cfg:
            section["scheduler"] = scheduler_cfg.get("type")
            section["warmup_ratio"] = scheduler_cfg.get("warmup_ratio", section.get("warmup_ratio"))
        return section

    def _build_fault_section(self, config_row, run_config, model_section) -> Dict[str, Any]:
        layer_idx = (
            config_row.get("layer_idx")
            if config_row.get("layer_idx") is not None
            else run_config.get("layer_idx")
        )
        parameters = (
            run_config.get("severity_params")
            if run_config.get("severity_params") is not None
            else config_row.get("severity_params")
        )
        layer_mapping = model_section.get("layer_mapping", {})
        target = None
        if layer_idx is not None:
            layer_key = f"L{layer_idx}"
            target = {
                "layer_index": layer_idx,
                "layer_name": layer_mapping.get(layer_key),
                "module": run_config.get("module"),
                "heads": run_config.get("heads"),
            }
        return {
            "is_faulty": bool(config_row.get("is_faulty")),
            "fault_category": config_row.get("fault_category"),
            "fault_subcategory": config_row.get("fault_subcategory"),
            "fault_id": run_config.get("fault_id") or config_row.get("fault_id"),
            "description": (run_config.get("description") or (config_row.get("metadata") or {}).get("description")),
            "target": target,
            "parameters": parameters or {},
        }

    def _build_labels_section(self, config_row, kill_row) -> Dict[str, Any]:
        structural_verified = bool(kill_row.get("structural_verified")) if kill_row else not bool(config_row.get("is_faulty"))
        if kill_row is not None:
            validation_status = "Killed" if kill_row.get("killed") else "Survived"
        elif config_row.get("status") == "failed":
            validation_status = "Failed"
        else:
            validation_status = "Pending"
        labels = {
            "validation_status": validation_status,
            "kill_function_pvalue": kill_row.get("p_value") if kill_row else None,
            "kill_decision_metric": kill_row.get("decision_metric") if kill_row else None,
            "structural_verified": structural_verified,
            "fault_category": config_row.get("fault_category"),
            "fault_subcategory": config_row.get("fault_subcategory"),
            "training_status": config_row.get("status"),
        }
        if kill_row and kill_row.get("details"):
            labels["kill_details"] = kill_row["details"]
        return labels

    def _build_epoch_rows(self, epoch_metrics, validation_metrics) -> List[Dict[str, Any]]:
        val_map = {entry.get("epoch"): entry for entry in validation_metrics}
        rows: List[Dict[str, Any]] = []
        for idx, epoch_entry in enumerate(epoch_metrics):
            epoch_idx = epoch_entry.get("epoch", idx)
            row = {"epoch": int(epoch_idx) + 1}
            for key, value in epoch_entry.items():
                if key == "epoch":
                    continue
                cleaned_key = key[:-5] if key.endswith("_mean") else key
                row[cleaned_key] = self._to_json_value(value)
            val_entry = val_map.get(epoch_idx)
            if val_entry:
                for key, value in val_entry.items():
                    if key == "epoch":
                        continue
                    row[key] = self._to_json_value(value)
            rows.append(row)
        return rows

    @staticmethod
    def _to_json_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _load_pipeline_configs(config_path: Optional[str]) -> Dict[str, Any]:
        if config_path and Path(config_path).exists():
            path = Path(config_path)
            suffix = path.suffix.lower()
            if suffix == ".json":
                with open(path, "r") as handle:
                    return json.load(handle)
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML required for YAML pipeline configs.") from exc
            with open(path, "r") as handle:
                return yaml.safe_load(handle)
        return {}

    @staticmethod
    def _build_config_lookup(config_source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        if not config_source:
            return lookup
        for section in ("baseline", "faults"):
            for entry in config_source.get(section, []):
                lookup[str(entry.get("config_id"))] = entry
        return lookup


def main():
    parser = argparse.ArgumentParser(description="Export encoder dataset to JSON")
    parser.add_argument("--master-config", default="config/master_config.yaml")
    parser.add_argument("--h5", default="results/metrics.h5")
    parser.add_argument("--db", default="results/dataset.db")
    parser.add_argument("--pipeline-config", default="config/pipeline_configs.json")
    parser.add_argument("--output", default="results/final_dataset/dataset.json")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    builder = JSONDatasetBuilder(
        master_config_path=args.master_config,
        h5_path=args.h5,
        db_path=args.db,
        pipeline_config_path=args.pipeline_config,
    )
    records = builder.build(args.output, indent=args.indent)
    print(f"Exported {len(records)} configurations to {args.output}")


if __name__ == "__main__":
    main()
