"""Configuration Manager for ABNN Encoder Fault Injection Pipeline."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HardwareConfig:
    platform: str
    device: str
    memory_limit_mb: Optional[int] = None
    memory_warning_mb: Optional[int] = None


@dataclass
class ModelConfig:
    name: str
    type: str
    max_length: int
    cache_dir: str


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    fp16: bool
    gradient_checkpointing: bool


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.config = self._load_config()
        self._validate_config()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self) -> None:
        required_sections = [
            'project', 'hardware', 'reproducibility', 'model',
            'dataset', 'training', 'faults', 'metrics',
            'kill_functions', 'pipeline', 'memory', 'validation'
        ]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        seeds = self.config['reproducibility'].get('seeds', [])
        if len(seeds) < 1:
            raise ValueError("At least one seed is required")
        if self.config['training']['batch_size'] < 1:
            raise ValueError("Batch size must be at least 1")
        if self.config['training']['epochs'] < 1:
            raise ValueError("Epochs must be at least 1")

    def _create_directories(self) -> None:
        directories = [
            self.config['model'].get('cache_dir'),
            self.config['dataset'].get('cache_dir'),
        ]
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

    def get_hardware_config(self) -> HardwareConfig:
        hw = self.config['hardware']
        return HardwareConfig(
            platform=hw['platform'],
            device=hw.get('device', 'cpu'),
            memory_limit_mb=hw.get('memory_limit_mb'),
            memory_warning_mb=hw.get('memory_warning_mb')
        )

    def get_model_config(self) -> ModelConfig:
        model = self.config['model']
        return ModelConfig(
            name=model['name'],
            type=model['type'],
            max_length=model['max_length'],
            cache_dir=model['cache_dir']
        )

    def get_training_config(self) -> TrainingConfig:
        train = self.config['training']
        return TrainingConfig(
            epochs=train['epochs'],
            batch_size=train['batch_size'],
            gradient_accumulation_steps=train['gradient_accumulation_steps'],
            effective_batch_size=train['effective_batch_size'],
            learning_rate=train['learning_rate'],
            weight_decay=train['weight_decay'],
            warmup_ratio=train['warmup_ratio'],
            max_grad_norm=train['max_grad_norm'],
            fp16=train['fp16'],
            gradient_checkpointing=train['gradient_checkpointing']
        )

    def get_seeds(self) -> List[int]:
        return self.config['reproducibility']['seeds']

    def get_device(self) -> str:
        return self.config['hardware']['device']

    def get_memory_limit(self) -> int:
        return self.config['hardware'].get('memory_limit_mb', 0)

    def get_fault_categories(self) -> List[Dict[str, Any]]:
        return self.config['faults']['categories']

    def get_metrics_config(self) -> Dict[str, Any]:
        return self.config['metrics']

    def get_kill_function_config(self) -> Dict[str, Any]:
        return self.config['kill_functions']

    def get_storage_config(self) -> Dict[str, Any]:
        return self.config.get('storage', {})

    def get_checkpoint_config(self) -> Dict[str, Any]:
        return self.config.get('checkpointing', {})

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})

    def is_deterministic(self) -> bool:
        return self.config['reproducibility']['deterministic']

    def get_total_configurations(self) -> int:
        return self.config['configurations']['total_expected']

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


def load_config(config_path: str = "config/master_config.yaml") -> ConfigManager:
    return ConfigManager(config_path)
