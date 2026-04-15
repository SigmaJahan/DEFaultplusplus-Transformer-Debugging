"""TrainingMetrics — loss, LR, memory, step time."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import MetricModule
from ..inspector import ModelInspector


class TrainingMetrics(MetricModule):
    """Collect loss, learning-rate, memory, and runtime stats."""

    def collect(
        self,
        *,
        loss: Any = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        step_time: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Loss
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                metrics['train_loss'] = float(loss.detach().item())
            else:
                metrics['train_loss'] = float(loss)
            metrics['loss'] = metrics['train_loss']

        # Learning rate
        if optimizer is not None:
            metrics['train_learning_rate'] = optimizer.param_groups[0]['lr']

        # Step time
        if step_time is not None and step_time > 0:
            metrics['runtime_step_time'] = float(step_time)
            metrics['runtime_steps_per_sec'] = 1.0 / step_time

        # Memory
        device = next(model.parameters()).device if model is not None else torch.device('cpu')
        if device.type == 'cuda':
            metrics['runtime_memory_alloc_mb'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            metrics['runtime_memory_reserved_mb'] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        elif device.type == 'mps':
            alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
            metrics['runtime_memory_alloc_mb'] = alloc
            metrics['runtime_memory_reserved_mb'] = alloc
        else:
            metrics['runtime_memory_alloc_mb'] = 0.0
            metrics['runtime_memory_reserved_mb'] = 0.0

        return metrics
