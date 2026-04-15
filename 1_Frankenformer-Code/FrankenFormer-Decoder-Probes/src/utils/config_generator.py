"""
Configuration Generator for ABNN Fault Injection Pipeline

Generates the complete fault configuration matrix based on Option A specification.
Fault configurations are replicated for each seed with early/mid/late layer sampling
plus severity sweeps for configurable faults.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from src.constants import BASE_LAYER_DEPTH

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


# Layer rotation groups (Early/Middle/Late strategy) anchored to a 6-layer template
BASE_LAYER_ANCHORS = {
    "group_1": [0, 2, 4],  # Early(0), Mid(2), Late(4)
    "group_2": [1, 3, 5],  # Early(1), Mid(3), Late(5)
    "group_3": [0, 3, 5],  # Early(0), Mid(3), Late(5)
    "group_4": [1, 2, 4],  # Early(1), Mid(2), Late(4)
    "group_5": [0, 2, 5],  # Early(0), Mid(2), Late(5)
}


def _scale_layer_index(anchor_idx: int, base_depth: int, target_depth: int) -> int:
    """Scale an anchor index from base_depth space to target_depth space."""
    if target_depth <= 1 or base_depth <= 1:
        return 0
    scaled = round(anchor_idx * (target_depth - 1) / (base_depth - 1))
    return int(max(0, min(target_depth - 1, scaled)))


def build_layer_groups(num_layers: int, base_depth: int = BASE_LAYER_DEPTH) -> Dict[str, List[int]]:
    """
    Build layer rotation groups scaled to the provided decoder depth.

    Args:
        num_layers: Target number of decoder layers.
        base_depth: Anchor depth used for rotation templates (default: BASE_LAYER_DEPTH).

    Returns:
        Mapping of group names to layer indices appropriate for the target depth.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    layer_groups: Dict[str, List[int]] = {}
    for group, anchors in BASE_LAYER_ANCHORS.items():
        mapped: List[int] = []
        for anchor in anchors:
            idx = _scale_layer_index(anchor, base_depth, num_layers)
            if idx not in mapped:  # preserve order while deduplicating
                mapped.append(idx)
        layer_groups[group] = mapped
    return layer_groups


DEFAULT_LAYER_GROUPS = build_layer_groups(BASE_LAYER_DEPTH)

# Fault definitions with group assignments
FAULT_DEFINITIONS = {
    # E1: Masking Faults (3 faults, 9 configs)
    "E1.1": {"name": "zero_mask", "category": "masking", "group": "group_1", "severity": None},
    "E1.2": {"name": "inverted_mask", "category": "masking", "group": "group_2", "severity": None},
    "E1.3": {"name": "wrong_mask_broadcast", "category": "masking", "group": "group_3", "severity": None},

    # E2: QKV Projection Faults (7 faults, 21 configs)
    "E2.1": {"name": "zero_query", "category": "qkv", "group": "group_1", "severity": None},
    "E2.2": {"name": "zero_key", "category": "qkv", "group": "group_2", "severity": None},
    "E2.3": {"name": "zero_value", "category": "qkv", "group": "group_3", "severity": None},
    "E2.4": {"name": "swapped_qk", "category": "qkv", "group": "group_4", "severity": None},
    "E2.5": {"name": "tie_heads", "category": "qkv", "group": "group_5", "severity": None},
    "E2.6": {"name": "wrong_head_dim", "category": "qkv", "group": "group_5", "severity": None},
    "E2.7": {"name": "freeze_qkv", "category": "qkv", "group": "group_5", "severity": None},

    # E3: Score/Softmax Faults (4 faults, 21 configs with severity)
    "E3.1": {"name": "missing_scaling", "category": "score", "group": "group_1", "severity": None},
    "E3.2": {"name": "wrong_scaling_factor", "category": "score", "group": "group_2",
             "severity": {"param": "wrong_factor", "values": [2.0, 0.5, 4.0]}},
    "E3.3": {"name": "misplaced_dropout", "category": "score", "group": "group_3",
             "severity": {"param": "dropout_p", "values": [0.1, 0.3]}},
    "E3.4": {"name": "unsafe_type_cast", "category": "score", "group": "group_4", "severity": None},

    # E4: Positional Encoding Faults (4 faults, 21 configs with severity)
    "E4.1": {"name": "off_by_one", "category": "positional", "group": "group_1",
             "severity": {"param": "shift", "values": [1, 3, 5]}},
    "E4.2": {"name": "truncate_positions", "category": "positional", "group": "group_2",
             "severity": {"param": "max_position", "values": [100, 256]}},
    "E4.3": {"name": "double_position", "category": "positional", "group": "group_3", "severity": None},
    "E4.4": {"name": "missing_positional", "category": "positional", "group": "group_4", "severity": None},

    # E5: Kernel/Integration Faults (3 faults, 12 configs with severity)
    "E5.1": {"name": "force_unoptimized", "category": "kernel", "group": "group_1", "severity": None},
    "E5.2": {"name": "wrong_layout", "category": "kernel", "group": "group_2", "severity": None},
    "E5.3": {"name": "inconsistent_dropout", "category": "kernel", "group": "group_3",
             "severity": {"param": "wrong_dropout_p", "values": [0.3, 0.5]}},

    # E6: Variant Selection Faults (2 faults, 6 configs)
    "E6.1": {"name": "wrong_variant", "category": "variant", "group": "group_1", "severity": None},
    "E6.2": {"name": "causal_in_noncausal", "category": "variant", "group": "group_2", "severity": None},

    # E7: FFN Faults (3 faults)
    "E7.1": {"name": "ffn_weight_scaling", "category": "ffn", "group": "group_1",
             "severity": {"param": "alpha", "values": [0.9, 0.65, 0.3, 1.8]}},
    "E7.2": {"name": "ffn_neuron_drop", "category": "ffn", "group": "group_2",
             "severity": {"param": "drop_fraction", "values": [0.1, 0.3, 0.5]}},
    "E7.3": {"name": "activation_distortion", "category": "ffn", "group": "group_3",
             "severity": {"values": [
                 {"mode": "scaled_gelu", "scale": 1.1},
                 {"mode": "relu"},
                 {"mode": "sign"}
             ]}},

    # E8: LayerNorm Faults (3 faults)
    "E8.1": {"name": "ln_gamma_fault", "category": "layernorm", "group": "group_1",
             "severity": {"values": [
                 {"gamma_scale": 0.9},
                 {"gamma_scale": 0.6},
                 {"gamma_scale": 0.0, "reinitialize": True}
             ]}},
    "E8.2": {"name": "ln_beta_fault", "category": "layernorm", "group": "group_2",
             "severity": {"param": "delta_std", "values": [0.01, 0.05, 0.1]}},
    "E8.3": {"name": "ln_stats_fault", "category": "layernorm", "group": "group_3",
             "severity": {"values": [
                 {"mode": "eps_scale", "eps_scale": 2.0},
                 {"mode": "eps_scale", "eps_scale": 10.0},
                 {"mode": "force_unit_var", "eps_scale": 1.0}
             ]}},

    # E9: Residual Path Faults (3 faults)
    "E9.1": {"name": "residual_drop", "category": "residual", "group": "group_1",
             "severity": {"param": "target", "values": ["ffn", "attention", "both"]}},
    "E9.2": {"name": "residual_scale", "category": "residual", "group": "group_2",
             "severity": {"values": [
                 {"alpha": 0.9, "target": "ffn"},
                 {"alpha": 0.5, "target": "both"},
                 {"alpha": -1.0, "target": "both"}
             ]}},
    "E9.3": {"name": "residual_noise", "category": "residual", "group": "group_3",
             "severity": {"param": "noise_scale", "values": [0.01, 0.05, 0.1]}},

    # E10: Embedding Faults (3 faults)
    "E10.1": {"name": "embedding_zero", "category": "embedding", "group": "group_1",
              "severity": {"values": [
                  {"fraction": 0.02},
                  {"fraction": 0.08},
                  {"fraction": 0.2}
              ]}},
    "E10.2": {"name": "embedding_swap", "category": "embedding", "group": "group_2",
              "severity": {"param": "swaps", "values": [1, 3, 6]}},
    "E10.3": {"name": "type_embedding_drop", "category": "embedding", "group": "group_3",
              "severity": {"param": "scale", "values": [0.5, 0.0]}},

    # E11: Output Projection Faults (3 faults)
    "E11.1": {"name": "out_scale", "category": "output", "group": "group_1",
              "severity": {"param": "alpha", "values": [0.9, 0.6, 1.8]}},
    "E11.2": {"name": "out_row_drop", "category": "output", "group": "group_2",
              "severity": {"param": "drop_fraction", "values": [0.05, 0.15, 0.3]}},
    "E11.3": {"name": "out_noise", "category": "output", "group": "group_3",
              "severity": {"values": [
                  {"noise_std": 0.01},
                  {"noise_std": 0.05},
                  {"noise_std": 0.1, "reinit_fraction": 0.2}
              ]}},

    # E12: Decoder Masking Faults (3 faults - decoder-specific)
    "E12.1": {"name": "break_causal_mask", "category": "decoder_masking", "group": "group_1",
              "severity": {"param": "visibility_ratio", "values": [0.3, 0.5, 0.7]}},
    "E12.2": {"name": "over_mask_valid_tokens", "category": "decoder_masking", "group": "group_2",
              "severity": {"param": "mask_ratio", "values": [0.1, 0.3, 0.5]}},
    "E12.3": {"name": "pad_masking_error", "category": "decoder_masking", "group": "group_3",
              "severity": {"values": [
                  {"error_type": "allow_pad_attention", "batch_corruption": 0.3},
                  {"error_type": "mask_valid_tokens", "batch_corruption": 0.5}
              ]}},

    # E13: KV-Cache Faults (4 faults - decoder-specific)
    "E13.1": {"name": "stale_cache", "category": "kv_cache", "group": "group_1",
              "severity": {"param": "freeze_after", "values": [50, 100, 200]}},
    "E13.2": {"name": "off_by_one_index", "category": "kv_cache", "group": "group_2",
              "severity": {"values": [
                  {"offset": 1, "mode": "read"},
                  {"offset": -1, "mode": "write"}
              ]}},
    "E13.3": {"name": "truncated_cache", "category": "kv_cache", "group": "group_3",
              "severity": {"values": [
                  {"truncate_last": 5, "trigger_length": 100},
                  {"truncate_last": 10, "trigger_length": 200}
              ]}},
    "E13.4": {"name": "cross_request_cache_leak", "category": "kv_cache", "group": "group_4",
              "severity": {"param": "leak_ratio", "values": [0.3, 0.5, 0.8]}},
}

# Clean baseline seeds (multi-seed setup)
BASELINE_SEEDS = [42, 123, 456, 789, 101112]


def generate_fault_configurations(
    layer_groups: Optional[Dict[str, List[int]]] = None,
    base_depth: int = BASE_LAYER_DEPTH,
) -> List[Dict[str, Any]]:
    """
    Generate all fault configurations based on Option A specification.

    Returns:
        List of configuration dictionaries, each containing:
        - config_id: Unique identifier
        - fault_id: Fault identifier (e.g., "E1.1")
        - fault_name: Fault type name
        - category: Fault category
        - layer_idx: Target layer
        - severity_params: Optional severity parameters
        - is_baseline: False for faults
    """
    groups = layer_groups or DEFAULT_LAYER_GROUPS
    configurations: List[Dict[str, Any]] = []
    config_counter = 1

    # Generate faulty configurations replicated per seed
    for seed in BASELINE_SEEDS:
        for fault_id, fault_def in FAULT_DEFINITIONS.items():
            fault_name = fault_def["name"]
            category = fault_def["category"]
            group = fault_def["group"]
            layers = groups[group]
            severity_spec = fault_def["severity"]

            if severity_spec is None:
                # No severity variations - one config per layer
                for layer_idx in layers:
                    config = {
                        "config_id": config_counter,
                        "fault_id": fault_id,
                        "fault_name": fault_name,
                        "category": category,
                        "layer_idx": layer_idx,
                        "base_depth": base_depth,
                        "severity_params": None,
                        "is_baseline": False,
                        "seed": seed,
                        "description": f"{fault_id} ({fault_name}) at layer {layer_idx}, seed={seed}"
                    }
                    configurations.append(config)
                    config_counter += 1
            else:
                # Severity variations - multiple configs per layer
                param_name = severity_spec.get("param")
                param_values = severity_spec.get("values", [])
                dict_mode = param_values and isinstance(param_values[0], dict)

                for layer_idx in layers:
                    for param_value in param_values:
                        if dict_mode and isinstance(param_value, dict):
                            severity_params = dict(param_value)
                            desc_suffix = ", ".join(f"{k}={v}" for k, v in severity_params.items())
                        elif param_name is not None:
                            severity_params = {param_name: param_value}
                            desc_suffix = f"{param_name}={param_value}"
                        else:
                            severity_params = {}
                            desc_suffix = f"severity={param_value}"

                        config = {
                            "config_id": config_counter,
                            "fault_id": fault_id,
                            "fault_name": fault_name,
                            "category": category,
                            "layer_idx": layer_idx,
                            "base_depth": base_depth,
                            "severity_params": severity_params,
                            "is_baseline": False,
                            "seed": seed,
                            "description": f"{fault_id} ({fault_name}) at layer {layer_idx}, {desc_suffix}, seed={seed}"
                        }
                        configurations.append(config)
                        config_counter += 1

    return configurations


def generate_baseline_configurations(base_depth: int = BASE_LAYER_DEPTH) -> List[Dict[str, Any]]:
    """
    Generate clean baseline configurations.

    Returns:
        List of baseline configurations (one per seed)
    """
    configurations = []

    for i, seed in enumerate(BASELINE_SEEDS, start=1):
        config = {
            "config_id": f"baseline_{i}",
            "fault_id": None,
            "fault_name": "clean_baseline",
            "category": "baseline",
            "layer_idx": None,
            "base_depth": base_depth,
            "severity_params": None,
            "is_baseline": True,
            "seed": seed,
            "description": f"Clean baseline with seed {seed}"
        }
        configurations.append(config)

    return configurations


def generate_all_configurations(num_layers: int = BASE_LAYER_DEPTH) -> Dict[str, Any]:
    """
    Generate complete configuration set for Option A pipeline.

    Returns:
        Dictionary containing:
        - metadata: Pipeline configuration metadata
        - baseline: List of baseline configurations
        - faults: List of fault configurations
        - summary: Configuration summary statistics
    """
    layer_groups = build_layer_groups(num_layers)
    baseline_configs = generate_baseline_configurations(base_depth=num_layers)
    fault_configs = generate_fault_configurations(layer_groups=layer_groups, base_depth=num_layers)

    # Count configurations by category
    category_counts = {}
    severity_counts = 0

    for config in fault_configs:
        category = config["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        if config["severity_params"] is not None:
            severity_counts += 1

    # Build complete configuration set
    all_configs = {
        "metadata": {
            "pipeline_name": "ABNN Fault Injection - Decoder Models",
            "generation_date": "2026-01-12",
            "strategy": "Strategic Layer Sampling with Rotation",
            "total_faults": len(FAULT_DEFINITIONS),
            "decoder_specific_faults": 7,
            "encoder_agnostic_faults": 38,
            "layer_strategy": "Early/Middle/Late rotation (3 layers per fault)",
            "severity_variations": len([f for f in FAULT_DEFINITIONS.values() if f["severity"] is not None]),
            "random_seeds": BASELINE_SEEDS,
            "layer_depth": num_layers,
        },
        "baseline": baseline_configs,
        "faults": fault_configs,
        "summary": {
            "total_configurations": len(baseline_configs) + len(fault_configs),
            "baseline_configurations": len(baseline_configs),
            "fault_configurations": len(fault_configs),
            "base_fault_configs": len(fault_configs) - severity_counts,
            "severity_variation_configs": severity_counts,
            "configurations_by_category": category_counts,
            "layer_groups": layer_groups,
        }
    }

    return all_configs


def save_configurations(output_path: str, format: str = "yaml", num_layers: int = BASE_LAYER_DEPTH) -> None:
    """
    Generate and save configurations to file.

    Args:
        output_path: Path to save configuration file
        format: Output format ("yaml" or "json")
        num_layers: Target decoder depth to scale layer groups
    """
    configs = generate_all_configurations(num_layers=num_layers)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "yaml":
        if yaml is None:
            raise RuntimeError("PyYAML is required to export YAML files. Install pyyaml or choose format='json'.")
        with open(output_file, "w") as f:
            yaml.dump(configs, f, default_flow_style=False, sort_keys=False)
    elif format == "json":
        with open(output_file, "w") as f:
            json.dump(configs, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Configuration saved to: {output_file}")
    print(f"  Total configurations: {configs['summary']['total_configurations']}")
    print(f"  - Baseline: {configs['summary']['baseline_configurations']}")
    print(f"  - Faults: {configs['summary']['fault_configurations']}")
    print(f"    - Base: {configs['summary']['base_fault_configs']}")
    print(f"    - Severity: {configs['summary']['severity_variation_configs']}")


def print_configuration_summary(num_layers: int = BASE_LAYER_DEPTH) -> None:
    """Print a summary of generated configurations."""
    configs = generate_all_configurations(num_layers=num_layers)

    print("=" * 80)
    print("ABNN Fault Injection Pipeline - Option A Configuration")
    print("=" * 80)
    print()
    print(f"Strategy: {configs['metadata']['strategy']}")
    print(f"Total Faults: {configs['metadata']['total_faults']}")
    print(f"Layer Strategy: {configs['metadata']['layer_strategy']}")
    print()
    print("Configuration Summary:")
    print(f"  Total Configurations: {configs['summary']['total_configurations']}")
    print(f"  - Clean Baseline: {configs['summary']['baseline_configurations']}")
    print(f"  - Faulty Variants: {configs['summary']['fault_configurations']}")
    print(f"    - Base configs: {configs['summary']['base_fault_configs']}")
    print(f"    - Severity variations: {configs['summary']['severity_variation_configs']}")
    print()
    print("Configurations by Category:")
    for category, count in configs['summary']['configurations_by_category'].items():
        print(f"  - {category}: {count}")
    print()
    print("Layer Groups:")
    for group, layers in configs['summary']['layer_groups'].items():
        print(f"  - {group}: {layers}")
    print()
    print("=" * 80)


def get_configuration_for_run(config_id: int, num_layers: int = BASE_LAYER_DEPTH) -> Dict[str, Any]:
    """
    Get a specific configuration by ID for pipeline execution.

    Args:
        config_id: Configuration ID to retrieve

    Returns:
        Configuration dictionary for the specified ID
    """
    all_configs = generate_all_configurations(num_layers=num_layers)

    # Check baseline first
    for config in all_configs["baseline"]:
        if config["config_id"] == f"baseline_{config_id}":
            return config

    # Check faults
    for config in all_configs["faults"]:
        if config["config_id"] == config_id:
            return config

    raise ValueError(f"Configuration ID {config_id} not found")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ABNN fault injection configurations for decoder models")
    parser.add_argument(
        "--output",
        default="config/pipeline_configs.json",
        help="Output file path (default: config/pipeline_configs.json)"
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary without saving"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=BASE_LAYER_DEPTH,
        help=f"Target decoder depth to scale layer groups (default: {BASE_LAYER_DEPTH})"
    )

    args = parser.parse_args()

    if args.summary_only:
        print_configuration_summary(num_layers=args.num_layers)
    else:
        print_configuration_summary(num_layers=args.num_layers)
        print()
        save_configurations(args.output, args.format, num_layers=args.num_layers)
