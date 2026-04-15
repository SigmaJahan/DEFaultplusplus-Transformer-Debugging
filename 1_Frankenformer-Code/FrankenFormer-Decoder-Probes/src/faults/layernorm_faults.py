"""
LayerNorm Faults (Group 3)

Implements:
- LNGammaFault: scale or zero gamma
- LNBetaFault: shift beta
- LNStatsFault: corrupt epsilon/statistics
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn

from src.faults.base_fault import BaseFault


class _LayerNormFault(BaseFault):
    """Helper base for accessing LayerNorm modules."""

    def _get_layer_norms(self) -> List[Tuple[str, nn.LayerNorm]]:
        """
        Get all LayerNorm modules from the layer.

        Supports both encoder (BERT, DistilBERT, RoBERTa) and decoder (GPT-2, DistilGPT2) architectures.

        Returns:
            List of (name, LayerNorm) tuples
        """
        layer = self.get_layer()
        norms: List[Tuple[str, nn.LayerNorm]] = []

        # GPT-2/Decoder-style norms (ln_1 before attention, ln_2 before MLP)
        if hasattr(layer, 'ln_1') and isinstance(layer.ln_1, nn.LayerNorm):
            norms.append(('ln_1', layer.ln_1))
        if hasattr(layer, 'ln_2') and isinstance(layer.ln_2, nn.LayerNorm):
            norms.append(('ln_2', layer.ln_2))

        # DistilBERT-style norms
        if hasattr(layer, 'sa_layer_norm') and isinstance(layer.sa_layer_norm, nn.LayerNorm):
            norms.append(('sa_layer_norm', layer.sa_layer_norm))
        if hasattr(layer, 'output_layer_norm') and isinstance(layer.output_layer_norm, nn.LayerNorm):
            norms.append(('output_layer_norm', layer.output_layer_norm))

        # BERT/RoBERTa-style norms
        attn_output = getattr(getattr(layer, 'attention', None), 'output', None)
        if attn_output is not None and hasattr(attn_output, 'LayerNorm'):
            if isinstance(attn_output.LayerNorm, nn.LayerNorm):
                norms.append(('attention_ln', attn_output.LayerNorm))
        output_ln = getattr(getattr(layer, 'output', None), 'LayerNorm', None)
        if isinstance(output_ln, nn.LayerNorm):
            norms.append(('output_ln', output_ln))

        # Generic LayerNorm attribute fallback
        if hasattr(layer, 'LayerNorm') and isinstance(getattr(layer, 'LayerNorm'), nn.LayerNorm):
            norms.append(('layer_norm', layer.LayerNorm))

        if not norms:
            raise ValueError(f"No LayerNorm modules found in layer {self.layer_idx}")
        return norms

    def _select_targets(self, target: str) -> List[Tuple[str, nn.LayerNorm]]:
        targets = self._get_layer_norms()
        if target == "all":
            return targets
        return [item for item in targets if item[0] == target]


class LNGammaFault(_LayerNormFault):
    """Modify gamma parameters (scaling/zeroing/reinit)."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        gamma_scale: float = 1.0,
        target: str = "all",
        reinitialize: bool = False
    ):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="ln_gamma_fault",
            description=f"Scale LayerNorm gamma by {gamma_scale}"
        )
        self.gamma_scale = float(gamma_scale)
        self.target = target
        self.reinitialize = bool(reinitialize)

    def inject(self) -> None:
        if self.is_injected:
            return

        targets = self._select_targets(self.target)
        self.original_state = {}
        for name, ln in targets:
            self.original_state[name] = ln.weight.detach().clone()
            with torch.no_grad():
                if self.reinitialize:
                    ln.weight.copy_(torch.randn_like(ln.weight))
                else:
                    ln.weight.mul_(self.gamma_scale)

        self.is_injected = True
        self._update_fault_metadata(
            "ln_gamma_fault",
            {"gamma_scale": self.gamma_scale, "target": self.target, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx} (target={self.target})")

    def restore(self) -> None:
        if not self.is_injected:
            return

        try:
            targets = self._select_targets(self.target)
        except Exception:
            targets = []

        for name, ln in targets:
            if name in self.original_state:
                with torch.no_grad():
                    ln.weight.copy_(self.original_state[name])

        self.original_state = {}
        self.is_injected = False
        self._update_fault_metadata("ln_gamma_fault", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class LNBetaFault(_LayerNormFault):
    """Shift beta parameters."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        delta_std: float = 0.01,
        target: str = "all"
    ):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="ln_beta_fault",
            description=f"Shift LayerNorm beta by N(0, {delta_std})"
        )
        self.delta_std = float(delta_std)
        self.target = target

    def inject(self) -> None:
        if self.is_injected:
            return

        targets = self._select_targets(self.target)
        self.original_state = {}
        for name, ln in targets:
            self.original_state[name] = ln.bias.detach().clone()
            noise = torch.randn_like(ln.bias) * self.delta_std
            with torch.no_grad():
                ln.bias.add_(noise)

        self.is_injected = True
        self._update_fault_metadata(
            "ln_beta_fault",
            {"delta_std": self.delta_std, "target": self.target, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx} (target={self.target})")

    def restore(self) -> None:
        if not self.is_injected:
            return

        try:
            targets = self._select_targets(self.target)
        except Exception:
            targets = []

        for name, ln in targets:
            if name in self.original_state:
                with torch.no_grad():
                    ln.bias.copy_(self.original_state[name])

        self.original_state = {}
        self.is_injected = False
        self._update_fault_metadata("ln_beta_fault", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class LNStatsFault(_LayerNormFault):
    """Corrupt epsilon or normalization statistics."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        eps_scale: float = 1.0,
        mode: str = "eps_scale",
        target: str = "all"
    ):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="ln_stats_fault",
            description=f"Alter LayerNorm stats mode={mode}"
        )
        self.eps_scale = float(eps_scale)
        self.mode = mode
        self.target = target
        self.original_eps: Dict[str, float] = {}
        self.original_forward: Dict[str, Any] = {}

    def inject(self) -> None:
        if self.is_injected:
            return

        targets = self._select_targets(self.target)
        self.original_eps = {}
        self.original_forward = {}

        for name, ln in targets:
            self.original_eps[name] = getattr(ln, 'eps', 1e-12)
            self.original_forward[name] = ln.forward

            if self.mode == "eps_scale":
                ln.eps = self.original_eps[name] * self.eps_scale
            else:
                eps_value = self.original_eps[name] * self.eps_scale

                def faulty_forward(x, ln_module=ln, eps=eps_value):
                    mean = x.mean(dim=-1, keepdim=True)
                    if self.mode == "force_unit_var":
                        var = torch.ones_like(mean)
                    elif self.mode == "fixed_var":
                        var = torch.full_like(mean, fill_value=0.25)
                    else:
                        var = x.var(dim=-1, unbiased=False, keepdim=True)
                    return (x - mean) / torch.sqrt(var + eps) * ln_module.weight + ln_module.bias

                ln.forward = faulty_forward

        self.is_injected = True
        self._update_fault_metadata(
            "ln_stats_fault",
            {"mode": self.mode, "eps_scale": self.eps_scale, "target": self.target, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx} (mode={self.mode})")

    def restore(self) -> None:
        if not self.is_injected:
            return

        try:
            targets = self._select_targets(self.target)
        except Exception:
            targets = []

        for name, ln in targets:
            if name in self.original_eps:
                ln.eps = self.original_eps[name]
            if name in self.original_forward:
                ln.forward = self.original_forward[name]

        self.original_eps = {}
        self.original_forward = {}
        self.is_injected = False
        self._update_fault_metadata("ln_stats_fault", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


LAYER_NORM_FAULTS: Dict[str, Any] = {
    "ln_gamma_fault": LNGammaFault,
    "ln_beta_fault": LNBetaFault,
    "ln_stats_fault": LNStatsFault,
}


def create_layernorm_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int,
    **kwargs
) -> BaseFault:
    """Factory for LayerNorm faults."""
    if fault_type not in LAYER_NORM_FAULTS:
        raise ValueError(f"Unknown LayerNorm fault type: {fault_type}")
    return LAYER_NORM_FAULTS[fault_type](model, layer_idx, **kwargs)
