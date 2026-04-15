"""
Base model class for ABNN Encoder Fault Injection Dataset.

Provides abstract interface for model wrappers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        device: torch.device,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.original_forward = None

    @abstractmethod
    def load_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_attention_modules(self) -> Dict[str, nn.Module]:
        pass

    def forward(self, **inputs) -> Any:
        return self.model(**inputs)

    def __call__(self, **inputs) -> Any:
        return self.forward(**inputs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels
        }, path)

    def load_model_from_file(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_memory_usage(self) -> Dict[str, float]:
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        elif self.device.type == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024 / 1024
            reserved = allocated
        else:
            allocated = 0
            reserved = 0

        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "model_size_mb": model_size
        }

    def count_parameters(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }

    def enable_gradient_checkpointing(self):
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def get_model_info(self) -> Dict[str, Any]:
        params = self.count_parameters()
        memory = self.get_memory_usage()
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "device": str(self.device),
            "total_parameters": params["total"],
            "trainable_parameters": params["trainable"],
            "model_size_mb": memory["model_size_mb"],
            "memory_allocated_mb": memory["allocated_mb"]
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"


class FaultInjectorMixin:
    def __init__(self):
        self.fault_active = False
        self.fault_info = {}
        self.backup_state = {}

    def backup_module(self, module_name: str, module: nn.Module):
        self.backup_state[module_name] = {
            'state_dict': {k: v.detach().clone() for k, v in module.state_dict().items()},
            'forward': module.forward
        }

    def restore_module(self, module_name: str, module: nn.Module):
        if module_name in self.backup_state:
            module.load_state_dict(self.backup_state[module_name]['state_dict'])
            module.forward = self.backup_state[module_name]['forward']

    def clear_backup(self):
        self.backup_state.clear()
        self.fault_active = False
        self.fault_info.clear()

    def get_fault_info(self) -> Dict[str, Any]:
        return {
            "fault_active": self.fault_active,
            **self.fault_info
        }
