#!/usr/bin/env python3
"""Validation and smoke test for encoder pipeline components."""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_encoder_model
from src.faults import ALL_FAULTS
from src.faults import (
    create_masking_fault, create_qkv_fault, create_score_fault,
    create_positional_fault, create_kernel_fault, create_variant_fault,
    create_embedding_fault, create_ffn_fault, create_layernorm_fault,
    create_residual_fault, create_output_fault, create_pooler_fault,
)
from src.kill_functions.kill_criteria import create_kill_criteria
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase
from src.metrics.metric_collector import MetricCollector
from src.constants import ENCODER_FAULT_CATEGORIES


CATEGORY_FACTORIES = {
    "masking": create_masking_fault,
    "qkv": create_qkv_fault,
    "score": create_score_fault,
    "positional": create_positional_fault,
    "kernel": create_kernel_fault,
    "variant": create_variant_fault,
    "embedding": create_embedding_fault,
    "ffn": create_ffn_fault,
    "layernorm": create_layernorm_fault,
    "residual": create_residual_fault,
    "output": create_output_fault,
    "pooler": create_pooler_fault,
}

CATEGORY_SAMPLE_TYPES = {
    "masking": "zero_mask",
    "qkv": "zero_query",
    "score": "missing_scaling",
    "positional": "missing",
    "kernel": "force_unoptimized",
    "variant": "causal_in_bidirectional",
    "embedding": "zero",
    "ffn": "weight_scaling",
    "layernorm": "gamma_scale",
    "residual": "drop",
    "output": "scale",
    "pooler": "scale",
}

CATEGORY_SAMPLE_PARAMS = {
    "embedding": {"fraction": 0.1},
    "ffn": {"alpha": 0.5},
    "layernorm": {"gamma": 0.5},
    "output": {"alpha": 0.5},
    "pooler": {"alpha": 0.5},
}


def test_fault_injection():
    """Test inject/restore for all 12 encoder categories."""
    print("=== Test: Fault Injection (12 encoder categories) ===")
    device = torch.device("cpu")
    model_wrapper = load_encoder_model("distilbert-base-uncased", num_labels=2, device=device)

    passed, failed = 0, 0
    for category in ENCODER_FAULT_CATEGORIES:
        fault_type = CATEGORY_SAMPLE_TYPES.get(category)
        if fault_type is None:
            print(f"  SKIP {category}: no sample type defined")
            continue
        try:
            factory = CATEGORY_FACTORIES[category]
            fault = factory(fault_type)
            params = CATEGORY_SAMPLE_PARAMS.get(category, {})
            fault.inject(model_wrapper.model, layer_idx=0, **params)
            fault.restore(model_wrapper.model)
            print(f"  PASS {category}/{fault_type}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {category}/{fault_type}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"  Results: {passed} passed, {failed} failed\n")
    del model_wrapper
    return failed == 0


def test_feature_extraction():
    """Test structural + attention probe feature extraction."""
    print("=== Test: Feature Extraction (structural + attention probes) ===")
    device = torch.device("cpu")
    model_wrapper = load_encoder_model("distilbert-base-uncased", num_labels=2, device=device)

    tokenizer = model_wrapper.tokenizer
    inputs = tokenizer("This is a test sentence.", return_tensors="pt", padding="max_length", max_length=32)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model_wrapper.forward_with_attention(**inputs)
    assert outputs.hidden_states is not None, "hidden_states missing"
    assert outputs.attentions is not None, "attentions missing"
    print(f"  hidden_states layers: {len(outputs.hidden_states)}")
    print(f"  attention layers: {len(outputs.attentions)}")

    collector = MetricCollector(
        device=device,
        collect_per_batch=True,
        collect_attention=True,
        config={"num_hidden_layers": model_wrapper.get_num_layers()},
    )

    dummy_labels = torch.zeros(1, dtype=torch.long, device=device)
    metrics = collector.collect_batch_metrics(
        loss=0.5, model=model_wrapper.model, optimizer=None,
        outputs=outputs, labels=dummy_labels, batch_idx=0, epoch=0,
        batch=inputs,
    )

    structural_keys = [k for k in metrics if "ffn_delta" in k or "ln_std" in k or "residual_cos" in k]
    attention_keys = [k for k in metrics if "entropy" in k or "head_similarity" in k or "pre_softmax" in k]
    print(f"  structural probe keys: {len(structural_keys)}")
    print(f"  attention probe keys: {len(attention_keys)}")
    print(f"  total metrics collected: {len(metrics)}")
    print("  PASS\n")

    del model_wrapper
    return True


def test_kill_criteria_routing():
    """Test kill criteria creation for all 12 encoder categories."""
    print("=== Test: Kill Criteria Routing (12 categories) ===")
    passed, failed = 0, 0

    for category in ENCODER_FAULT_CATEGORIES:
        try:
            criteria = create_kill_criteria(category, f"test_{category}")
            dummy_clean = {"val_loss": [0.5], "val_accuracy": [0.9]}
            dummy_faulty = {"val_loss": [1.5], "val_accuracy": [0.3]}
            result = criteria.evaluate(dummy_clean, dummy_faulty, structural_check=True)
            assert "killed" in result, f"missing 'killed' key for {category}"
            print(f"  PASS {category}: killed={result['killed']}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {category}: {e}")
            failed += 1

    print(f"  Results: {passed} passed, {failed} failed\n")
    return failed == 0


def test_storage_roundtrip():
    """Test HDF5 and SQLite storage round-trip."""
    print("=== Test: Storage Round-Trip (HDF5 + SQLite) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test_metrics.h5")
        db_path = os.path.join(tmpdir, "test_dataset.db")

        h5 = HDF5MetricsStorage(h5_path, enable_locking=False)
        h5.save_configuration_metrics(
            config_id="test_001",
            epoch_metrics=[{"loss": 0.5, "accuracy": 0.8}],
            final_metrics={"val_accuracy": 0.85},
            metadata={"model": "distilbert"},
        )
        loaded = h5.load_configuration_metrics("test_001")
        assert loaded is not None, "HDF5 load returned None"
        assert loaded["final_metrics"]["val_accuracy"] == 0.85
        configs = h5.list_configurations()
        assert "test_001" in configs
        print("  PASS HDF5 round-trip")

        db = SQLiteDatabase(db_path)
        db.insert_configuration(
            config_id="test_001", seed=42,
            fault_category="masking", fault_subcategory="zero_mask",
            is_faulty=True, status="completed",
            model_name="distilbert-base-uncased", dataset_name="sst2",
        )
        db.update_configuration_results(
            config_id="test_001",
            final_accuracy=0.85, final_loss=0.4, final_f1_score=0.83,
            best_accuracy=0.87, best_loss=0.38, status="completed",
        )
        row = db.get_configuration("test_001")
        assert row is not None, "SQLite get returned None"
        assert row["final_accuracy"] == 0.85
        print("  PASS SQLite round-trip")

    print()
    return True


def test_config_validation():
    """Test config file loading and basic validation."""
    print("=== Test: Config File Validation ===")
    config_dir = ROOT / "config"

    matrix_path = config_dir / "matrix_encoder.yaml"
    if matrix_path.exists():
        with open(matrix_path) as f:
            matrix = yaml.safe_load(f)
        assert "models" in matrix, "matrix missing 'models'"
        assert "tasks" in matrix, "matrix missing 'tasks'"
        assert "training" in matrix, "matrix missing 'training'"
        print(f"  PASS matrix_encoder.yaml ({len(matrix['models'])} models, {len(matrix['tasks'])} tasks)")
    else:
        print("  SKIP matrix_encoder.yaml not found")

    pipeline_path = config_dir / "pipeline_configs_probes.json"
    if pipeline_path.exists():
        with open(pipeline_path) as f:
            configs = json.load(f)
        assert isinstance(configs, list), "pipeline configs not a list"
        assert len(configs) > 0, "pipeline configs empty"
        required_keys = {"config_id", "model_name", "task_name"}
        for cfg in configs:
            missing = required_keys - set(cfg.keys())
            assert not missing, f"config {cfg.get('config_id')} missing keys: {missing}"
        print(f"  PASS pipeline_configs_probes.json ({len(configs)} configs)")
    else:
        print("  SKIP pipeline_configs_probes.json not found")

    severity_path = config_dir / "silent_faults_severity.json"
    if severity_path.exists():
        with open(severity_path) as f:
            severity = json.load(f)
        faults = severity.get("fault_configs", [])
        assert len(faults) > 0, "no fault configs"
        print(f"  PASS silent_faults_severity.json ({len(faults)} fault configs)")
    else:
        print("  SKIP silent_faults_severity.json not found")

    print()
    return True


def main():
    print("FrankenFormer Encoder Pipeline Validation\n")
    results = {}

    results["config_validation"] = test_config_validation()
    results["storage_roundtrip"] = test_storage_roundtrip()
    results["kill_criteria"] = test_kill_criteria_routing()
    results["fault_injection"] = test_fault_injection()
    results["feature_extraction"] = test_feature_extraction()

    print("=" * 50)
    print("Summary:")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
