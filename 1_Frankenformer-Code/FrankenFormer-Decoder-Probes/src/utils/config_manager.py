"""
Configuration Manager for ABNN Fault Injection Dataset.

Handles loading, validation, and access to configuration parameters.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Hardware configuration settings."""
    platform: str
    device: str
    memory_limit_mb: Optional[int] = None
    memory_warning_mb: Optional[int] = None


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    type: str
    max_length: int
    cache_dir: str


@dataclass
class TrainingConfig:
    """Training configuration settings."""
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
    """
    Manages configuration loading and access.

    Provides structured access to all configuration parameters with validation.
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to master configuration YAML file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config = self._load_config()
        self._validate_config()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary containing all configuration parameters
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_sections = [
            'project', 'hardware', 'reproducibility', 'model',
            'dataset', 'training', 'faults', 'metrics',
            'kill_functions', 'pipeline', 'memory', 'validation'
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate seeds
        seeds = self.config['reproducibility'].get('seeds', [])
        if len(seeds) < 1:
            raise ValueError("At least one seed is required")

        # Validate batch size
        if self.config['training']['batch_size'] < 1:
            raise ValueError("Batch size must be at least 1")

        # Validate epochs
        if self.config['training']['epochs'] < 1:
            raise ValueError("Epochs must be at least 1")

    def _create_directories(self) -> None:
        """Create cache directories if provided."""
        directories = [
            self.config['model'].get('cache_dir'),
            self.config['dataset'].get('cache_dir'),
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

    # Hardware configuration
    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration."""
        hw = self.config['hardware']
        return HardwareConfig(
            platform=hw['platform'],
            device=hw.get('device', 'cpu'),
            memory_limit_mb=hw.get('memory_limit_mb'),
            memory_warning_mb=hw.get('memory_warning_mb')
        )

    # Model configuration
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model = self.config['model']
        return ModelConfig(
            name=model['name'],
            type=model['type'],
            max_length=model['max_length'],
            cache_dir=model['cache_dir']
        )

    # Training configuration
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
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

    # Convenience methods
    def get_seeds(self) -> List[int]:
        """Get list of random seeds."""
        return self.config['reproducibility']['seeds']

    def get_device(self) -> str:
        """Get device string for PyTorch."""
        return self.config['hardware']['device']

    def get_memory_limit(self) -> int:
        """Get memory limit in MB."""
        return self.config['hardware'].get('memory_limit_mb', 0)

    def get_fault_categories(self) -> List[Dict[str, Any]]:
        """Get all fault categories and subcategories."""
        return self.config['faults']['categories']

    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics collection configuration."""
        return self.config['metrics']

    def get_kill_function_config(self) -> Dict[str, Any]:
        """Get kill function configuration."""
        return self.config['kill_functions']

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.config.get('storage', {})

    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpointing configuration."""
        return self.config.get('checkpointing', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})

    def is_deterministic(self) -> bool:
        """Check if deterministic mode is enabled."""
        return self.config['reproducibility']['deterministic']

    def get_total_configurations(self) -> int:
        """Get total number of expected configurations."""
        return self.config['configurations']['total_expected']

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('training.batch_size')
            8
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-style access to top-level config sections.

        Args:
            key: Top-level configuration section name

        Returns:
            Configuration section
        """
        return self.config[key]

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigManager(config_path='{self.config_path}')"


def load_config(config_path: str = "config/master_config.yaml") -> ConfigManager:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)

