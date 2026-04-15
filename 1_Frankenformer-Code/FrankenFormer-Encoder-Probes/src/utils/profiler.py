"""GPU Profiling Utility for Memory and Performance Tracking."""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional
import torch

logger = logging.getLogger(__name__)


@dataclass
class ProfileMetrics:
    model_key: str
    task_key: str
    total_configs: int
    peak_memory_mb: float
    avg_time_per_config_sec: float
    gpu_name: str
    gpu_memory_total_mb: float
    timestamp: str
    batch_size: int
    max_length: int
    num_epochs: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProfileMetrics':
        return cls(**data)


class GPUProfiler:
    """Track GPU memory usage and timing for configurations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory_bytes = 0
        self.config_count = 0
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_name = "CPU"
            self.gpu_total_memory = 0

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def record_config_start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def record_config_end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_peak = torch.cuda.max_memory_allocated()
            if current_peak > self.peak_memory_bytes:
                self.peak_memory_bytes = current_peak
        self.config_count += 1

    def end(self) -> Dict:
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        avg_time_per_config = total_time / max(self.config_count, 1)
        peak_memory_mb = self.peak_memory_bytes / (1024 ** 2)
        total_memory_mb = self.gpu_total_memory / (1024 ** 2)
        return {
            'total_configs': self.config_count,
            'total_time_sec': total_time,
            'avg_time_per_config_sec': avg_time_per_config,
            'peak_memory_mb': peak_memory_mb,
            'gpu_name': self.gpu_name,
            'gpu_memory_total_mb': total_memory_mb,
        }


class ProfileStorage:
    """Store and retrieve profiling data for model-dataset-hardware combinations."""

    def __init__(self, config_path: str = "config/profiles.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles = self._load()

    def _load(self) -> Dict:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)

    def _get_key(self, model_key: str, task_key: str, gpu_name: str) -> str:
        gpu_short = gpu_name.lower().split()[0] if gpu_name else "unknown"
        return f"{model_key}:{task_key}:{gpu_short}"

    def save_profile(self, metrics: ProfileMetrics):
        key = self._get_key(metrics.model_key, metrics.task_key, metrics.gpu_name)
        self.profiles[key] = metrics.to_dict()
        self._save()

    def get_profile(self, model_key: str, task_key: str, gpu_name: Optional[str] = None) -> Optional[ProfileMetrics]:
        if gpu_name is None and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        elif gpu_name is None:
            return None
        key = self._get_key(model_key, task_key, gpu_name)
        if key in self.profiles:
            return ProfileMetrics.from_dict(self.profiles[key])
        return None

    def has_profile(self, model_key: str, task_key: str, gpu_name: Optional[str] = None) -> bool:
        return self.get_profile(model_key, task_key, gpu_name) is not None

    def list_profiles(self) -> Dict:
        return self.profiles.copy()


def calculate_parallel_config(
    profile: ProfileMetrics,
    available_memory_mb: Optional[float] = None,
    memory_buffer: float = 0.85,
    time_buffer: float = 1.3
) -> Dict:
    if available_memory_mb is None:
        available_memory_mb = profile.gpu_memory_total_mb
    usable_memory = available_memory_mb * memory_buffer
    configs_per_gpu = max(1, int(usable_memory / profile.peak_memory_mb))
    total_configs = profile.total_configs
    array_size = (total_configs + configs_per_gpu - 1) // configs_per_gpu
    time_per_batch = profile.avg_time_per_config_sec * configs_per_gpu
    total_time_parallel = time_per_batch * array_size
    total_time_sequential = profile.avg_time_per_config_sec * total_configs
    conservative_time_per_job_sec = time_per_batch * time_buffer
    slurm_time_hours = max(1, int((conservative_time_per_job_sec / 3600) + 0.999))
    return {
        'configs_per_gpu': configs_per_gpu,
        'array_size': array_size,
        'estimated_time_per_job_sec': time_per_batch,
        'estimated_total_time_parallel_sec': total_time_parallel,
        'estimated_total_time_sequential_sec': total_time_sequential,
        'estimated_speedup': total_time_sequential / max(total_time_parallel, 1),
        'peak_memory_per_config_mb': profile.peak_memory_mb,
        'total_memory_required_mb': profile.peak_memory_mb * configs_per_gpu,
        'available_memory_mb': available_memory_mb,
        'slurm_time_hours': slurm_time_hours,
        'slurm_time_str': f"{slurm_time_hours}:00:00",
    }
