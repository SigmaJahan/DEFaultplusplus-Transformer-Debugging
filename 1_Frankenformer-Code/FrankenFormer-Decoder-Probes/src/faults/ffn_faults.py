"""
Feed-Forward Network Faults (Group 2)

Implements three FFN fault families that operate on transformer blocks:
- FFNWeightScalingFault: scale W1/W2 by a factor alpha
- FFNNeuronDropFault: zero out a fraction of hidden neurons
- ActivationDistortionFault: swap or distort the FFN activation
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.faults.base_fault import BaseFault


class _ScaledGELU(nn.Module):
    """Module wrapper that scales inputs before GELU to keep nn.Module semantics."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x * self.scale)


class _ClippedGELU(nn.Module):
    """Module wrapper that clips GELU outputs to a fixed magnitude."""

    def __init__(self, clip_value: float):
        super().__init__()
        self.clip_value = float(clip_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.gelu(x), -self.clip_value, self.clip_value)


class _FFNFault(BaseFault):
    """Helper base to access FFN modules safely."""

    def _get_ffn(self) -> nn.Module:
        """
        Get FFN module for both encoder and decoder architectures.

        Returns a module with .lin1 and .lin2 attributes for consistency.
        """
        layer = self.get_layer()

        # DistilBERT-style: layer.ffn.lin1 / lin2
        ffn = getattr(layer, 'ffn', None)
        if ffn is not None and hasattr(ffn, 'lin1') and hasattr(ffn, 'lin2'):
            return ffn

        # GPT-2/Decoder-style: layer.mlp.c_fc / c_proj
        mlp = getattr(layer, 'mlp', None)
        if mlp is not None and hasattr(mlp, 'c_fc') and hasattr(mlp, 'c_proj'):
            # Create adapter to provide consistent lin1/lin2 interface
            class _FFNAdapter(nn.Module):
                def __init__(self, lin1: nn.Module, lin2: nn.Module, activation: nn.Module):
                    super().__init__()
                    self.lin1 = lin1
                    self.lin2 = lin2
                    self.activation = activation
            # GPT-2 uses 'act' for activation (typically NewGELU or GELU)
            activation = getattr(mlp, 'act', None) or getattr(mlp, 'activation', None) or nn.GELU()
            return _FFNAdapter(mlp.c_fc, mlp.c_proj, activation)

        # BERT/RoBERTa-style: intermediate.dense and output.dense
        if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense') \
           and hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
            class _FFNAdapter(nn.Module):
                def __init__(self, lin1: nn.Module, lin2: nn.Module, activation: nn.Module):
                    super().__init__()
                    self.lin1 = lin1
                    self.lin2 = lin2
                    self.activation = activation
            # BERT uses 'intermediate_act_fn' for activation
            activation = getattr(layer.intermediate, 'intermediate_act_fn', None) or nn.GELU()
            return _FFNAdapter(layer.intermediate.dense, layer.output.dense, activation)

        raise ValueError(f"Layer {self.layer_idx} has no FFN module")


class FFNWeightScalingFault(_FFNFault):
    """
    Multiply W1/W2 (and biases) by a scalar alpha to simulate scaling faults.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        alpha: float = 1.0,
        scale_w1: bool = True,
        scale_w2: bool = True,
    ):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="ffn_weight_scaling",
            description=f"Scale FFN weights by alpha={alpha}"
        )
        self.alpha = float(alpha)
        self.scale_w1 = bool(scale_w1)
        self.scale_w2 = bool(scale_w2)

    def inject(self) -> None:
        if self.is_injected:
            return

        ffn = self._get_ffn()
        self.target_layer = self.get_layer()

        # Backup parameters
        self.original_state = {
            "lin1_weight": ffn.lin1.weight.detach().clone(),
            "lin2_weight": ffn.lin2.weight.detach().clone(),
        }
        if ffn.lin1.bias is not None:
            self.original_state["lin1_bias"] = ffn.lin1.bias.detach().clone()
        if ffn.lin2.bias is not None:
            self.original_state["lin2_bias"] = ffn.lin2.bias.detach().clone()

        with torch.no_grad():
            if self.scale_w1:
                ffn.lin1.weight.mul_(self.alpha)
                if ffn.lin1.bias is not None:
                    ffn.lin1.bias.mul_(self.alpha)
            if self.scale_w2:
                ffn.lin2.weight.mul_(self.alpha)
                if ffn.lin2.bias is not None:
                    ffn.lin2.bias.mul_(self.alpha)

        self.is_injected = True
        self._update_fault_metadata(
            "ffn_weight_scaling",
            {"alpha": self.alpha, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} (alpha={self.alpha}) into layer {self.layer_idx}")

    def restore(self) -> None:
        if not self.is_injected:
            return

        try:
            ffn = self._get_ffn()
        except Exception:
            ffn = None

        if ffn is not None:
            with torch.no_grad():
                ffn.lin1.weight.copy_(self.original_state["lin1_weight"])
                ffn.lin2.weight.copy_(self.original_state["lin2_weight"])
                if ffn.lin1.bias is not None and "lin1_bias" in self.original_state:
                    ffn.lin1.bias.copy_(self.original_state["lin1_bias"])
                if ffn.lin2.bias is not None and "lin2_bias" in self.original_state:
                    ffn.lin2.bias.copy_(self.original_state["lin2_bias"])

        self.original_state = {}
        self.is_injected = False
        self._update_fault_metadata("ffn_weight_scaling", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class FFNNeuronDropFault(_FFNFault):
    """
    Zero out a fraction p of hidden neurons to simulate dead FFN units.
    """

    def __init__(self, model: nn.Module, layer_idx: int, drop_fraction: float = 0.1):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="ffn_neuron_drop",
            description=f"Zero {drop_fraction:.2f} fraction of FFN hidden neurons"
        )
        self.drop_fraction = float(drop_fraction)
        self.drop_indices: Optional[torch.Tensor] = None

    def inject(self) -> None:
        if self.is_injected:
            return

        ffn = self._get_ffn()
        self.target_layer = self.get_layer()

        hidden_dim = ffn.lin1.weight.size(0)
        num_drop = max(1, int(hidden_dim * self.drop_fraction))

        gen = torch.Generator()
        seed_val = int(self.layer_idx * 1000 + num_drop)
        gen.manual_seed(seed_val)
        perm = torch.randperm(hidden_dim, generator=gen)
        drop_indices = perm[:num_drop]
        self.drop_indices = drop_indices

        self.original_state = {
            "lin1_weight": ffn.lin1.weight.detach().clone(),
            "lin2_weight": ffn.lin2.weight.detach().clone()
        }
        if ffn.lin1.bias is not None:
            self.original_state["lin1_bias"] = ffn.lin1.bias.detach().clone()
        if ffn.lin2.bias is not None:
            self.original_state["lin2_bias"] = ffn.lin2.bias.detach().clone()

        with torch.no_grad():
            mask_lin1 = torch.ones_like(ffn.lin1.weight)
            mask_lin1[drop_indices] = 0
            ffn.lin1.weight.mul_(mask_lin1)
            if ffn.lin1.bias is not None:
                ffn.lin1.bias[drop_indices] = 0

            mask_lin2 = torch.ones_like(ffn.lin2.weight)
            mask_lin2[:, drop_indices] = 0
            ffn.lin2.weight.mul_(mask_lin2)
            # lin2.bias is per output dim; keep intact to avoid introducing bias drift

        self.is_injected = True
        self._update_fault_metadata(
            "ffn_neuron_drop",
            {"drop_fraction": self.drop_fraction, "layer": self.layer_idx, "indices": drop_indices.tolist()}
        )
        self._log_info(f"Injected {self.fault_name} (p={self.drop_fraction:.2f}) into layer {self.layer_idx}")

    def restore(self) -> None:
        if not self.is_injected:
            return

        try:
            ffn = self._get_ffn()
        except Exception:
            ffn = None

        if ffn is not None:
            with torch.no_grad():
                ffn.lin1.weight.copy_(self.original_state["lin1_weight"])
                ffn.lin2.weight.copy_(self.original_state["lin2_weight"])
                if ffn.lin1.bias is not None and "lin1_bias" in self.original_state:
                    ffn.lin1.bias.copy_(self.original_state["lin1_bias"])
                if ffn.lin2.bias is not None and "lin2_bias" in self.original_state:
                    ffn.lin2.bias.copy_(self.original_state["lin2_bias"])

        self.original_state = {}
        self.drop_indices = None
        self.is_injected = False
        self._update_fault_metadata("ffn_neuron_drop", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class ActivationDistortionFault(_FFNFault):
    """
    Distort the FFN activation function (scale, swap, or clip).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        mode: str = "scaled_gelu",
        scale: float = 1.05,
        clip_value: float = 0.5
    ):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="activation_distortion",
            description=f"FFN activation distortion ({mode})"
        )
        self.mode = mode
        self.scale = float(scale)
        self.clip_value = float(clip_value)
        self.original_activation = None

    def inject(self) -> None:
        if self.is_injected:
            return

        ffn = self._get_ffn()
        self.target_layer = self.get_layer()

        self.original_activation = ffn.activation

        if self.mode == "relu":
            faulty_activation = nn.ReLU()
        elif self.mode == "scaled_gelu":
            faulty_activation = _ScaledGELU(self.scale)
        elif self.mode == "clipped":
            faulty_activation = _ClippedGELU(self.clip_value)
        elif self.mode == "sign":
            faulty_activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation distortion mode: {self.mode}")

        ffn.activation = faulty_activation
        self.is_injected = True
        self._update_fault_metadata(
            "activation_distortion",
            {"mode": self.mode, "scale": self.scale, "clip": self.clip_value, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} ({self.mode}) into layer {self.layer_idx}")

    def restore(self) -> None:
        if not self.is_injected:
            return

        try:
            ffn = self._get_ffn()
        except Exception:
            ffn = None

        if ffn is not None and self.original_activation is not None:
            ffn.activation = self.original_activation

        self.original_activation = None
        self.is_injected = False
        self._update_fault_metadata("activation_distortion", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


# Registry / factory
FFN_FAULTS: Dict[str, Any] = {
    "ffn_weight_scaling": FFNWeightScalingFault,
    "ffn_neuron_drop": FFNNeuronDropFault,
    "activation_distortion": ActivationDistortionFault,
}


def create_ffn_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int,
    **kwargs
) -> BaseFault:
    """
    Factory to create FFN faults.

    Args:
        fault_type: One of ffn_weight_scaling, ffn_neuron_drop, activation_distortion
        model: Target model
        layer_idx: Transformer block index
        **kwargs: Fault-specific args
    """
    if fault_type not in FFN_FAULTS:
        raise ValueError(f"Unknown FFN fault type: {fault_type}")
    fault_cls = FFN_FAULTS[fault_type]
    return fault_cls(model, layer_idx, **kwargs)
