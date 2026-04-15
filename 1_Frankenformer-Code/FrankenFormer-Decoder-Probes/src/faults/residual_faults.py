"""
Residual Path Faults (Group 4)

Implements:
- ResidualDropFault: remove residual contribution
- ResidualScaleFault: scale/invert residual branch
- ResidualNoiseFault: inject Gaussian noise into residual path
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from src.faults.base_fault import BaseFault


def _get_dropout(layer: nn.Module, candidates: Optional[list] = None) -> Optional[nn.Module]:
    candidates = candidates or ["dropout_sa", "sa_dropout", "dropout", "dropout_ffn"]
    for name in candidates:
        if hasattr(layer, name):
            module = getattr(layer, name)
            if isinstance(module, nn.Module):
                return module
    return None


class _ResidualFault(BaseFault):
    """Shared helper to patch transformer block forward for residual path changes."""

    def _build_faulty_forward(
        self,
        attn_alpha: float = 1.0,
        ffn_alpha: float = 1.0,
        drop_attention_residual: bool = False,
        drop_ffn_residual: bool = False,
        noise_scale: float = 0.0,
        residual_floor: float = 0.1,
        original_forward=None,
    ):
        layer = self.get_layer()

        # Get attention module (encoder: .attention, decoder: .attn)
        attention = getattr(layer, 'attention', None)
        if attention is None:
            attention = getattr(layer, 'attn', None)  # GPT-2/decoder style

        # Build FFN callable across architectures
        ffn = getattr(layer, 'ffn', None)
        if ffn is None:
            # GPT-2/Decoder style: layer.mlp
            ffn = getattr(layer, 'mlp', None)

        if ffn is None:
            # BERT/RoBERTa style: intermediate + output (without built-in residual)
            if hasattr(layer, 'intermediate') and hasattr(layer, 'output'):
                def ffn(x):
                    dense = layer.intermediate(x)
                    out = layer.output.dense(dense)
                    if hasattr(layer.output, 'dropout'):
                        out = layer.output.dropout(out)
                    return out

        if attention is None or ffn is None:
            raise ValueError(f"Layer {self.layer_idx} missing attention or FFN module")

        sa_ln = getattr(layer, 'sa_layer_norm', None) or getattr(layer, 'layer_norm1', None)
        ffn_ln = getattr(layer, 'output_layer_norm', None) or getattr(layer, 'layer_norm2', None)
        attn_dropout = _get_dropout(layer, ["dropout_sa", "sa_dropout", "dropout"])
        ffn_dropout = _get_dropout(layer, ["dropout_ffn", "dropout"])

        # DistilBERT (TransformerBlock) wrapper: fall back to original forward and
        # adjust the combined output to avoid signature differences that led to
        # shape issues during attention.
        if layer.__class__.__name__ == "TransformerBlock" and original_forward is not None:
            def distilbert_forward(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                if isinstance(outputs, tuple):
                    hidden = outputs[0]
                    attn_weights = outputs[1] if len(outputs) > 1 else None
                else:
                    hidden = outputs
                    attn_weights = None

                combined = hidden
                if drop_attention_residual or drop_ffn_residual:
                    combined = residual_floor * hidden
                else:
                    combined = ffn_alpha * hidden if drop_ffn_residual else hidden * ffn_alpha

                if noise_scale > 0.0:
                    ref = hidden.detach()
                    ref_norm = ref.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
                    combined = combined + torch.randn_like(hidden) * noise_scale * (ref_norm + 1e-6)

                return (combined, attn_weights)

            return distilbert_forward

        def faulty_forward(
            *args,
            **kwargs
        ):
            # Handle possible binding of `layer` as the first positional arg
            arg0_is_layer = len(args) > 0 and isinstance(args[0], nn.Module)
            start = 1 if arg0_is_layer else 0

            # Normalize inputs across DistilBERT (qkv) and BERT-style signatures
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > start:
                hidden_states = args[start]
            # Save original residual input
            x = hidden_states
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is None and len(args) > start + 1:
                attention_mask = args[start + 1]
            head_mask = kwargs.get("head_mask", None)
            if head_mask is None and len(args) > start + 2:
                head_mask = args[start + 2]
            encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
            encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
            past_key_value = kwargs.get("past_key_value", None)
            layer_past = kwargs.get("layer_past", None)
            use_cache = kwargs.get("use_cache", False)
            cache_position = kwargs.get("cache_position", None)
            output_attentions = kwargs.get("output_attentions", False)
            if not output_attentions and len(args) > start + 3 and isinstance(args[start + 3], bool):
                output_attentions = args[start + 3]

            if layer_past is None and len(args) > start + 1 and isinstance(args[start + 1], (tuple, list)):
                layer_past = args[start + 1]
                if attention_mask is None and len(args) > start + 2:
                    attention_mask = args[start + 2]
                if head_mask is None and len(args) > start + 3:
                    head_mask = args[start + 3]
                if len(args) > start + 4 and isinstance(args[start + 4], bool):
                    use_cache = args[start + 4]
                if len(args) > start + 5 and isinstance(args[start + 5], bool):
                    output_attentions = args[start + 5]
                if cache_position is None and len(args) > start + 6:
                    cache_position = args[start + 6]

            if hidden_states is None:
                raise ValueError("Residual fault forward missing hidden_states")

            # Handle different attention module signatures across architectures
            if hasattr(attention, 'self'):
                # BERT-style: BertSelfAttention wrapped in BertAttention
                attn_outputs = attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            elif hasattr(layer, 'attn'):
                # GPT2-style: GPT2Attention (decoder-only, uses attention_mask not mask)
                attn_outputs = attention(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )
            else:
                # DistilBERT-style: MultiHeadSelfAttention uses qkv signature
                attn_outputs = attention(
                    hidden_states, hidden_states, hidden_states,
                    mask=attention_mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions
                )
            if isinstance(attn_outputs, (tuple, list)):
                attn_context = attn_outputs[0]
                attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
            else:
                attn_context = attn_outputs
                attn_weights = None

            if attn_dropout is not None:
                attn_context = attn_dropout(attn_context)

            if drop_attention_residual:
                # Keep a small residual to avoid collapse/divergence
                attn_combined = attn_context + residual_floor * x
            else:
                attn_combined = attn_context + attn_alpha * x

            if sa_ln is not None:
                attn_normed = sa_ln(attn_combined)
            else:
                attn_normed = attn_combined

            ffn_out = ffn(attn_normed)
            if ffn_dropout is not None:
                ffn_out = ffn_dropout(ffn_out)

            if noise_scale > 0.0:
                ref = attn_normed.detach()
                ref_norm = ref.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
                noise = torch.randn_like(ffn_out) * noise_scale * (ref_norm + 1e-6)
                ffn_out = ffn_out + noise

            if drop_ffn_residual:
                combined = ffn_out + residual_floor * attn_normed
            else:
                combined = ffn_out + ffn_alpha * attn_normed

            if ffn_ln is not None:
                combined = ffn_ln(combined)

            if use_cache:
                cached_value = None
                if isinstance(attn_outputs, (tuple, list)) and len(attn_outputs) > 1:
                    cached_value = attn_outputs[1]
                if output_attentions:
                    return (combined, cached_value, attn_weights)
                return (combined, cached_value)

            if output_attentions:
                return (combined, attn_weights)
            return (combined,)

        return faulty_forward


class ResidualDropFault(_ResidualFault):
    """Remove residual contribution around attention and/or FFN."""

    def __init__(self, model: nn.Module, layer_idx: int, target: str = "ffn"):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="residual_drop",
            description=f"Remove residual path ({target})"
        )
        self.target = target
        self.original_forward = None

    def inject(self) -> None:
        if self.is_injected:
            return
        layer = self.get_layer()
        self.target_layer = layer
        self.original_forward = layer.forward

        drop_attention = self.target in ("attention", "both")
        drop_ffn = self.target in ("ffn", "both")
        layer.forward = self._build_faulty_forward(
            attn_alpha=1.0,
            ffn_alpha=1.0,
            drop_attention_residual=drop_attention,
            drop_ffn_residual=drop_ffn,
            noise_scale=0.0,
            original_forward=self.original_forward,
        )
        self.is_injected = True
        self._update_fault_metadata(
            "residual_drop",
            {"target": self.target, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx} (target={self.target})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        layer = self.get_layer()
        if self.original_forward is not None:
            layer.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._update_fault_metadata("residual_drop", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class ResidualScaleFault(_ResidualFault):
    """Scale or invert residual contribution."""

    def __init__(self, model: nn.Module, layer_idx: int, alpha: float = 0.8, target: str = "ffn"):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="residual_scale",
            description=f"Scale residual contribution by {alpha}"
        )
        self.alpha = float(alpha)
        self.target = target
        self.original_forward = None

    def inject(self) -> None:
        if self.is_injected:
            return
        layer = self.get_layer()
        self.target_layer = layer
        self.original_forward = layer.forward

        # Clamp scaling to keep training stable
        safe_alpha = max(0.3, min(self.alpha, 1.5))
        attn_alpha = safe_alpha if self.target in ("attention", "both") else 1.0
        ffn_alpha = safe_alpha if self.target in ("ffn", "both") else 1.0

        layer.forward = self._build_faulty_forward(
            attn_alpha=attn_alpha,
            ffn_alpha=ffn_alpha,
            drop_attention_residual=False,
            drop_ffn_residual=False,
            noise_scale=0.0,
            original_forward=self.original_forward,
        )
        self.is_injected = True
        self._update_fault_metadata(
            "residual_scale",
            {"alpha": self.alpha, "target": self.target, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx} (alpha={self.alpha}, target={self.target})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        layer = self.get_layer()
        if self.original_forward is not None:
            layer.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._update_fault_metadata("residual_scale", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class ResidualNoiseFault(_ResidualFault):
    """Inject Gaussian noise into the residual path."""

    def __init__(self, model: nn.Module, layer_idx: int, noise_scale: float = 0.002, target: str = "ffn"):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="residual_noise",
            description=f"Add noise (scale={noise_scale}) to residual path"
        )
        self.noise_scale = float(noise_scale)
        self.target = target
        self.original_forward = None

    def inject(self) -> None:
        if self.is_injected:
            return
        layer = self.get_layer()
        self.target_layer = layer
        self.original_forward = layer.forward

        drop_attention = False
        drop_ffn = False
        attn_alpha = 1.0
        ffn_alpha = 1.0
        if self.target == "attention":
            # noise applied after attention residual combination only
            ffn_alpha = 1.0
        elif self.target == "both":
            # leave alphas at 1, noise added to final residual combination
            pass

        layer.forward = self._build_faulty_forward(
            attn_alpha=attn_alpha,
            ffn_alpha=ffn_alpha,
            drop_attention_residual=drop_attention,
            drop_ffn_residual=drop_ffn,
            noise_scale=self.noise_scale,
            original_forward=self.original_forward,
        )
        self.is_injected = True
        self._update_fault_metadata(
            "residual_noise",
            {"noise_scale": self.noise_scale, "target": self.target, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx} (scale={self.noise_scale})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        layer = self.get_layer()
        if self.original_forward is not None:
            layer.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._update_fault_metadata("residual_noise", None)
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


RESIDUAL_FAULTS: Dict[str, Any] = {
    "residual_drop": ResidualDropFault,
    "residual_scale": ResidualScaleFault,
    "residual_noise": ResidualNoiseFault,
}


def create_residual_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int,
    **kwargs
) -> BaseFault:
    """Factory for residual path faults."""
    if fault_type not in RESIDUAL_FAULTS:
        raise ValueError(f"Unknown residual fault type: {fault_type}")
    return RESIDUAL_FAULTS[fault_type](model, layer_idx, **kwargs)
