"""
Base model class for ABNN Fault Injection Dataset.

Provides abstract interface for model wrappers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.

    Provides common interface for model operations including
    fault injection, training, and evaluation.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        device: torch.device,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model wrapper.

        Args:
            model_name: Name of the pretrained model
            num_labels: Number of output labels
            device: Device to load model on
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.cache_dir = cache_dir

        # CRITICAL FIX: Do NOT call load_model() here
        # Subclasses will call it explicitly after setting up their own device and config
        # This prevents double-loading and allows device to be properly set before loading
        self.model = None
        self.original_forward = None

    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Load the model.

        Returns:
            PyTorch model
        """
        pass

    @abstractmethod
    def get_attention_modules(self) -> Dict[str, nn.Module]:
        """
        Get attention modules from the model.

        Returns:
            Dictionary mapping layer names to attention modules
        """
        pass

    def forward(self, **inputs) -> Any:
        """
        Forward pass through the model.

        Args:
            **inputs: Model inputs

        Returns:
            Model outputs
        """
        return self.model(**inputs)

    def __call__(self, **inputs) -> Any:
        """
        Call model directly.

        Args:
            **inputs: Model inputs

        Returns:
            Model outputs
        """
        return self.forward(**inputs)

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def to(self, device: torch.device):
        """
        Move model to device.

        Args:
            device: Target device
        """
        self.device = device
        self.model.to(device)
        return self

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()

    def state_dict(self):
        """Get model state dictionary."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load model state dictionary.

        Args:
            state_dict: State dictionary to load
        """
        self.model.load_state_dict(state_dict)

    def save_model(self, path: str):
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels
        }, path)

    def load_model_from_file(self, path: str):
        """
        Load model from file.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage of the model.

        Returns:
            Dictionary with memory statistics in MB
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        elif self.device.type == "mps":
            # MPS doesn't have detailed memory stats
            allocated = torch.mps.current_allocated_memory() / 1024 / 1024
            reserved = allocated  # Approximate
        else:
            allocated = 0
            reserved = 0

        # Estimate model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "model_size_mb": model_size
        }

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model information
        """
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
        """String representation."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"


class FaultInjectorMixin:
    """
    Mixin class for fault injection capabilities.

    Provides common fault injection utilities.
    """

    def __init__(self):
        """Initialize fault injector."""
        self.fault_active = False
        self.fault_info = {}
        self.backup_state = {}

    def backup_module(self, module_name: str, module: nn.Module):
        """
        Backup a module before fault injection.

        Args:
            module_name: Name of the module
            module: Module to backup
        """
        self.backup_state[module_name] = {
            # Clone tensors so mutations during faults do not leak into backups
            'state_dict': {k: v.detach().clone() for k, v in module.state_dict().items()},
            'forward': module.forward
        }

    def restore_module(self, module_name: str, module: nn.Module):
        """
        Restore a module after fault injection.

        Args:
            module_name: Name of the module
            module: Module to restore
        """
        if module_name in self.backup_state:
            module.load_state_dict(self.backup_state[module_name]['state_dict'])
            module.forward = self.backup_state[module_name]['forward']

    def clear_backup(self):
        """Clear all backed up state."""
        self.backup_state.clear()
        self.fault_active = False
        self.fault_info.clear()

    def get_fault_info(self) -> Dict[str, Any]:
        """
        Get information about active fault.

        Returns:
            Dictionary with fault information
        """
        return {
            "fault_active": self.fault_active,
            **self.fault_info
        }
