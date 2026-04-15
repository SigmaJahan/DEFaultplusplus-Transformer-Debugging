"""
Output Projection Faults (Group 6)

Operate on the final projection head (classifier or LM head).
- OutScaleFault: scale output weights
- OutRowDropFault: zero selected rows
- OutNoiseFault: add noise or reinitialize slices
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from src.faults.base_fault import BaseFault


class _OutputFault(BaseFault):
    """Helper base to access output projection."""

    def _get_output_layer(self) -> nn.Module:
        # DistilBERT classifier head
        if hasattr(self.model, 'classifier'):
            cls = self.model.classifier
            # RobertaClassificationHead style
            if hasattr(cls, 'out_proj'):
                return cls.out_proj
            # Generic linear head
            if hasattr(cls, 'weight'):
                return cls
        # Decoder-only LM head
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        # Generic final linear
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == getattr(self.model.config, 'num_labels', module.out_features):
                return module
        raise ValueError("Could not locate output projection layer")


class OutScaleFault(_OutputFault):
    """Scale output projection weights."""

    def __init__(self, model: nn.Module, layer_idx: int = -1, alpha: float = 1.0):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="out_scale",
            description=f"Scale output projection by {alpha}"
        )
        self.alpha = float(alpha)

    def inject(self) -> None:
        if self.is_injected:
            return
        layer = self._get_output_layer()
        self.target_layer = layer
        self.original_state = {
            "weight": layer.weight.detach().clone()
        }
        if layer.bias is not None:
            self.original_state["bias"] = layer.bias.detach().clone()

        with torch.no_grad():
            layer.weight.mul_(self.alpha)
            if layer.bias is not None:
                layer.bias.mul_(self.alpha)

        self.is_injected = True
        self._update_fault_metadata(
            "out_scale",
            {"alpha": self.alpha}
        )
        self._log_info(f"Injected {self.fault_name} (alpha={self.alpha})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        try:
            layer = self._get_output_layer()
            with torch.no_grad():
                layer.weight.copy_(self.original_state["weight"])
                if layer.bias is not None and "bias" in self.original_state:
                    layer.bias.copy_(self.original_state["bias"])
        except Exception:
            pass
        self.original_state = {}
        self.is_injected = False
        self._update_fault_metadata("out_scale", None)
        self._log_info(f"Restored from {self.fault_name}")


class OutRowDropFault(_OutputFault):
    """Zero selected rows (vocab/class indices) of output projection."""

    def __init__(self, model: nn.Module, layer_idx: int = -1, drop_fraction: float = 0.1):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="out_row_drop",
            description=f"Zero {drop_fraction:.2f} of output rows"
        )
        self.drop_fraction = float(drop_fraction)
        self.indices: Optional[torch.Tensor] = None

    def inject(self) -> None:
        if self.is_injected:
            return
        layer = self._get_output_layer()
        self.target_layer = layer

        rows = layer.weight.size(0)
        count = max(1, int(rows * self.drop_fraction))
        gen = torch.Generator()
        gen.manual_seed(int(self.drop_fraction * 1000) + rows)
        perm = torch.randperm(rows, generator=gen)
        indices = perm[:count]
        self.indices = indices

        self.original_state = {
            "weight": layer.weight.detach().clone()
        }
        if layer.bias is not None:
            self.original_state["bias"] = layer.bias.detach().clone()

        with torch.no_grad():
            layer.weight[indices] = 0
            if layer.bias is not None:
                layer.bias[indices] = 0

        self.is_injected = True
        self._update_fault_metadata(
            "out_row_drop",
            {"indices": indices.tolist(), "fraction": self.drop_fraction}
        )
        self._log_info(f"Injected {self.fault_name} (rows={len(indices)})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        try:
            layer = self._get_output_layer()
            with torch.no_grad():
                layer.weight.copy_(self.original_state["weight"])
                if layer.bias is not None and "bias" in self.original_state:
                    layer.bias.copy_(self.original_state["bias"])
        except Exception:
            pass
        self.original_state = {}
        self.indices = None
        self.is_injected = False
        self._update_fault_metadata("out_row_drop", None)
        self._log_info(f"Restored from {self.fault_name}")


class OutNoiseFault(_OutputFault):
    """Add noise or reinitialize parts of the output projection."""

    def __init__(self, model: nn.Module, layer_idx: int = -1, noise_std: float = 0.01, reinit_fraction: float = 0.0):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="out_noise",
            description=f"Add noise (std={noise_std}) to output projection"
        )
        self.noise_std = float(noise_std)
        self.reinit_fraction = float(reinit_fraction)
        self.original_state = {}

    def inject(self) -> None:
        if self.is_injected:
            return
        layer = self._get_output_layer()
        self.target_layer = layer
        rows = layer.weight.size(0)
        count = max(0, int(rows * self.reinit_fraction))

        gen = torch.Generator()
        gen.manual_seed(int(self.noise_std * 1e4) + rows)
        perm = torch.randperm(rows, generator=gen)
        reinit_rows = perm[:count] if count > 0 else torch.tensor([], dtype=torch.long)

        self.original_state = {
            "weight": layer.weight.detach().clone()
        }
        if layer.bias is not None:
            self.original_state["bias"] = layer.bias.detach().clone()

        with torch.no_grad():
            layer.weight.add_(torch.randn_like(layer.weight) * self.noise_std)
            if layer.bias is not None:
                layer.bias.add_(torch.randn_like(layer.bias) * self.noise_std)
            if count > 0:
                layer.weight[reinit_rows] = torch.randn_like(layer.weight[reinit_rows])
                if layer.bias is not None:
                    layer.bias[reinit_rows] = 0

        self.is_injected = True
        self._update_fault_metadata(
            "out_noise",
            {"noise_std": self.noise_std, "reinit_fraction": self.reinit_fraction}
        )
        self._log_info(f"Injected {self.fault_name} (noise_std={self.noise_std}, reinit_rows={count})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        try:
            layer = self._get_output_layer()
            with torch.no_grad():
                layer.weight.copy_(self.original_state["weight"])
                if layer.bias is not None and "bias" in self.original_state:
                    layer.bias.copy_(self.original_state["bias"])
        except Exception:
            pass
        self.original_state = {}
        self.is_injected = False
        self._update_fault_metadata("out_noise", None)
        self._log_info(f"Restored from {self.fault_name}")


OUTPUT_FAULTS: Dict[str, Any] = {
    "out_scale": OutScaleFault,
    "out_row_drop": OutRowDropFault,
    "out_noise": OutNoiseFault,
}


def create_output_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int = -1,
    **kwargs
) -> BaseFault:
    """Factory for output projection faults."""
    if fault_type not in OUTPUT_FAULTS:
        raise ValueError(f"Unknown output projection fault type: {fault_type}")
    return OUTPUT_FAULTS[fault_type](model, layer_idx, **kwargs)
