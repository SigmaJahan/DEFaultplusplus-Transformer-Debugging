"""
GPU Profiling Utility for Memory and Performance Tracking

This module provides profiling capabilities to collect GPU memory usage,
execution time, and configuration counts for model-dataset pairs.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import torch


logger = logging.getLogger(__name__)


@dataclass
class ProfileMetrics:
    """Metrics collected during profiling run."""

    model_key: str
    task_key: str
    total_configs: int
    peak_memory_mb: float
    avg_time_per_config_sec: float
    gpu_name: str
    gpu_memory_total_mb: float
    timestamp: str

    # Additional metadata
    batch_size: int
    max_length: int
    num_epochs: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProfileMetrics':
        """Create from dictionary."""
        return cls(**data)


class GPUProfiler:
    """Track GPU memory usage and timing for configurations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory_bytes = 0
        self.config_count = 0

        # GPU info
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_name = "CPU"
            self.gpu_total_memory = 0

    def start(self):
        """Start profiling session."""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        logger.info(f"Profiling started on {self.gpu_name}")

    def record_config_start(self):
        """Record start of a configuration run."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def record_config_end(self):
        """Record end of a configuration run and update peak memory."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_peak = torch.cuda.max_memory_allocated()
            if current_peak > self.peak_memory_bytes:
                self.peak_memory_bytes = current_peak

        self.config_count += 1

    def end(self) -> Dict:
        """End profiling and return metrics."""
        self.end_time = time.time()

        total_time = self.end_time - self.start_time
        avg_time_per_config = total_time / max(self.config_count, 1)
        peak_memory_mb = self.peak_memory_bytes / (1024 ** 2)
        total_memory_mb = self.gpu_total_memory / (1024 ** 2)

        metrics = {
            'total_configs': self.config_count,
            'total_time_sec': total_time,
            'avg_time_per_config_sec': avg_time_per_config,
            'peak_memory_mb': peak_memory_mb,
            'gpu_name': self.gpu_name,
            'gpu_memory_total_mb': total_memory_mb,
        }

        logger.info(f"Profiling complete: {self.config_count} configs")
        logger.info(f"Peak memory: {peak_memory_mb:.2f} MB")
        logger.info(f"Avg time per config: {avg_time_per_config:.2f} sec")

        return metrics


class ProfileStorage:
    """Store and retrieve profiling data for model-dataset-hardware combinations."""

    def __init__(self, config_path: str = "config/profiles.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles = self._load()

    def _load(self) -> Dict:
        """Load existing profiles from disk."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        """Save profiles to disk."""
        with open(self.config_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)

    def _get_key(self, model_key: str, task_key: str, gpu_name: str) -> str:
        """Generate unique key for model-dataset-hardware combination."""
        # Normalize GPU name (e.g., "NVIDIA A100-SXM4-80GB" -> "a100")
        gpu_short = gpu_name.lower().split()[0] if gpu_name else "unknown"
        return f"{model_key}:{task_key}:{gpu_short}"

    def save_profile(self, metrics: ProfileMetrics):
        """Save profile metrics for a model-dataset-hardware combination."""
        key = self._get_key(metrics.model_key, metrics.task_key, metrics.gpu_name)
        self.profiles[key] = metrics.to_dict()
        self._save()
        logger.info(f"Saved profile: {key}")

    def get_profile(self, model_key: str, task_key: str, gpu_name: Optional[str] = None) -> Optional[ProfileMetrics]:
        """Retrieve profile metrics for a model-dataset-hardware combination."""
        if gpu_name is None and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        elif gpu_name is None:
            return None

        key = self._get_key(model_key, task_key, gpu_name)

        if key in self.profiles:
            logger.info(f"Found existing profile: {key}")
            return ProfileMetrics.from_dict(self.profiles[key])

        logger.info(f"No profile found for: {key}")
        return None

    def has_profile(self, model_key: str, task_key: str, gpu_name: Optional[str] = None) -> bool:
        """Check if profile exists for model-dataset-hardware combination."""
        return self.get_profile(model_key, task_key, gpu_name) is not None

    def list_profiles(self) -> Dict:
        """List all available profiles."""
        return self.profiles.copy()


def calculate_parallel_config(
    profile: ProfileMetrics,
    available_memory_mb: Optional[float] = None,
    memory_buffer: float = 0.85,
    time_buffer: float = 1.3
) -> Dict:
    """
    Calculate optimal parallel execution parameters based on profile.

    Args:
        profile: Profile metrics from test run
        available_memory_mb: Available GPU memory (uses profile if not provided)
        memory_buffer: Safety factor for memory allocation (default 0.85 = 85%)
        time_buffer: Safety factor for time allocation (default 1.3 = 30% overhead)

    Returns:
        Dictionary with parallel execution parameters
    """
    if available_memory_mb is None:
        available_memory_mb = profile.gpu_memory_total_mb

    # Calculate configs per GPU based on memory
    usable_memory = available_memory_mb * memory_buffer
    configs_per_gpu = int(usable_memory / profile.peak_memory_mb)

    # Must run at least 1 config
    configs_per_gpu = max(1, configs_per_gpu)

    # Calculate array job size
    total_configs = profile.total_configs
    array_size = (total_configs + configs_per_gpu - 1) // configs_per_gpu  # Ceiling division

    # Estimate time
    time_per_batch = profile.avg_time_per_config_sec * configs_per_gpu
    total_time_sequential = profile.avg_time_per_config_sec * total_configs
    total_time_parallel = time_per_batch * array_size
    speedup = total_time_sequential / max(total_time_parallel, 1)

    # Calculate conservative SLURM time allocation
    # Add time_buffer for overhead (data loading, initialization, cleanup)
    conservative_time_per_job_sec = time_per_batch * time_buffer

    # Round up to next hour for SLURM time format
    conservative_time_hours = int((conservative_time_per_job_sec / 3600) + 0.999)  # Ceiling
    conservative_time_hours = max(1, conservative_time_hours)  # At least 1 hour

    # Format as HH:MM:SS for SLURM
    slurm_time_hours = conservative_time_hours
    slurm_time_str = f"{slurm_time_hours}:00:00"

    return {
        'configs_per_gpu': configs_per_gpu,
        'array_size': array_size,
        'estimated_time_per_job_sec': time_per_batch,
        'estimated_total_time_parallel_sec': total_time_parallel,
        'estimated_total_time_sequential_sec': total_time_sequential,
        'estimated_speedup': speedup,
        'peak_memory_per_config_mb': profile.peak_memory_mb,
        'total_memory_required_mb': profile.peak_memory_mb * configs_per_gpu,
        'available_memory_mb': available_memory_mb,
        'memory_buffer': memory_buffer,
        'time_buffer': time_buffer,
        'conservative_time_per_job_sec': conservative_time_per_job_sec,
        'slurm_time_hours': slurm_time_hours,
        'slurm_time_str': slurm_time_str,
    }
