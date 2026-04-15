"""
Variant Faults (E6 Category)

Implements two types of attention variant selection faults:
- E6.1: Wrong Attention Variant (using simpler/wrong attention mechanism)
- E6.2: Causal in Non-Causal (applying causal masking to bidirectional encoder)

All faults target the variant/configuration of attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable
from src.faults.base_fault import AttentionFault

def _prepare_mask(mask, batch_size: int, n_heads: int, q_length: int, k_length: int, dtype: torch.dtype):
    """Broadcast attention mask to match attention scores shape."""
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(dtype=dtype)
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.expand(batch_size, n_heads, q_length, k_length)
    return mask


def _apply_head_mask(weights, head_mask: Optional[torch.Tensor]):
    """Apply head mask with safe broadcasting."""
    if head_mask is None:
        return weights
    if head_mask.dim() == 1:
        head_mask = head_mask.view(1, -1, 1, 1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
    return weights * head_mask


class WrongVariantFault(AttentionFault):
    """
    E6.1: Wrong Attention Variant

    Uses a simpler attention variant instead of multi-head attention.
    This simulates a bug where the wrong attention mechanism is selected.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize wrong variant fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="wrong_variant",
            description="Uses single-head attention instead of multi-head"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the wrong variant fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        # Create faulty forward that uses simplified single-head attention
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass using simplified single-head attention."""
            bs, q_length, dim = query.size()

            def _pad_context(context, pad_dim, num_heads, head_dim):
                if context.dim() == 4 and num_heads and head_dim:
                    head_pad = max(num_heads - 1, 0)
                    context = context[:, :1, :, :]
                    if head_pad:
                        padding = torch.zeros(
                            bs,
                            head_pad,
                            q_length,
                            head_dim,
                            dtype=context.dtype,
                            device=context.device,
                        )
                        context = torch.cat([context, padding], dim=1)
                    context = context.permute(0, 2, 1, 3).contiguous()
                    return context.view(bs, q_length, num_heads * head_dim)
                if context.dim() == 3 and pad_dim > 0:
                    padding = torch.zeros(
                        bs,
                        q_length,
                        pad_dim,
                        dtype=context.dtype,
                        device=context.device,
                    )
                    return torch.cat([context, padding], dim=-1)
                return context

            # Handle BERT-style attention
            if hasattr(attention, 'self'):
                self_attn = attention.self
                head_dim = self_attn.attention_head_size

                # Project Q, K, V
                q = self_attn.query(query)
                k = self_attn.key(key)
                v = self_attn.value(value)

                # FAULTY: Use only first head
                q = q[..., :head_dim]
                k = k[..., :head_dim]
                v = v[..., :head_dim]

                scores = torch.matmul(q, k.transpose(-2, -1))
                scores = scores / math.sqrt(head_dim)

                if mask is not None:
                    mask_2d = mask.squeeze(1).squeeze(1) if mask.dim() == 4 else mask.squeeze(1) if mask.dim() == 3 else mask
                    mask_2d = mask_2d.unsqueeze(1)
                    scores = scores + mask_2d

                weights = F.softmax(scores, dim=-1)
                weights = self_attn.dropout(weights)

                context = torch.matmul(weights, v)
                context = _pad_context(
                    context,
                    self_attn.all_head_size - head_dim,
                    getattr(self_attn, "num_attention_heads", 1),
                    head_dim,
                )

                context = attention.output.dense(context)
                context = attention.output.dropout(context)
                context = attention.output.LayerNorm(context + query)

                # Always return tuple (context, weights_or_none) for consistency
                return (context, weights.unsqueeze(1) if output_attentions else None)

            # Handle DistilBERT-style attention
            elif hasattr(attention, 'q_lin') and hasattr(attention, 'k_lin'):
                embed_dim = attention.dim or dim
                head_dim = embed_dim // max(attention.n_heads or 1, 1)

                q = attention.q_lin(query)
                k = attention.k_lin(key)
                v = attention.v_lin(value)

                # FAULTY: Use only first head
                q = q[..., :head_dim]
                k = k[..., :head_dim]
                v = v[..., :head_dim]

                scores = torch.matmul(q, k.transpose(-2, -1))
                scores = scores / math.sqrt(max(head_dim, 1))

                if mask is not None:
                    mask_2d = mask.squeeze(1).squeeze(1) if mask.dim() == 4 else mask.squeeze(1) if mask.dim() == 3 else mask
                    mask_2d = mask_2d.unsqueeze(1)
                    scores = scores + mask_2d

                weights = F.softmax(scores, dim=-1)

                if attention.training and attention.dropout is not None:
                    weights = attention.dropout(weights)

                context = torch.matmul(weights, v)
                context = _pad_context(
                    context,
                    embed_dim - head_dim,
                    attention.n_heads or 1,
                    head_dim,
                )

                context = attention.out_lin(context)

                # Always return tuple (context, weights_or_none) for consistency
                return (context, weights.unsqueeze(1) if output_attentions else None)

            # Handle GPT-2-style attention
            elif hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj'):
                embed_dim = getattr(attention, 'embed_dim', dim)
                num_heads = getattr(attention, 'num_heads', getattr(attention, 'n_heads', 1))
                head_dim = embed_dim // num_heads

                qkv = attention.c_attn(query)
                q, k, v = qkv.split(embed_dim, dim=2)

                # FAULTY: Use only first head
                q = q[..., :head_dim]
                k = k[..., :head_dim]
                v = v[..., :head_dim]

                scores = torch.matmul(q, k.transpose(-2, -1))
                scores = scores / math.sqrt(head_dim)

                if mask is not None:
                    mask_2d = mask.squeeze(1).squeeze(1) if mask.dim() == 4 else mask.squeeze(1) if mask.dim() == 3 else mask
                    mask_2d = mask_2d.unsqueeze(1)
                    scores = scores + mask_2d

                weights = F.softmax(scores, dim=-1)

                if hasattr(attention, 'attn_dropout') and attention.attn_dropout is not None:
                    weights = attention.attn_dropout(weights)

                context = torch.matmul(weights, v)
                context = _pad_context(
                    context,
                    embed_dim - head_dim,
                    num_heads,
                    head_dim,
                )

                context = attention.c_proj(context)

                if hasattr(attention, 'resid_dropout') and attention.resid_dropout is not None:
                    context = attention.resid_dropout(context)

                # Always return tuple (context, weights_or_none) for consistency
                return (context, weights.unsqueeze(1) if output_attentions else None)

            else:
                raise ValueError(f"Unsupported attention architecture: {type(attention)}")

        # Replace forward method
        if hasattr(attention, 'self'):
            attention.self.forward = self._build_attention_wrapper(faulty_forward)
        else:
            attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
            if hasattr(attention, 'self'):
                attention.self.forward = self._backup_forward
            else:
                attention.forward = self._backup_forward

        self.original_forward = None
        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class CausalInNonCausalFault(AttentionFault):
    """
    E6.2: Causal in Non-Causal

    Applies causal masking to bidirectional encoder attention.
    This simulates a bug where encoder uses causal attention (like decoder).
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize causal in non-causal fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="causal_in_noncausal",
            description="Applies causal masking to bidirectional encoder"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the causal in non-causal fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Detect if this is a decoder-only model (GPT-2)
        # For decoder models that already use causal attention, this fault is not applicable
        is_decoder_model = hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj')

        if is_decoder_model:
            # Skip injection for decoder-only models - they already use causal attention
            # Applying causal mask to GPT-2 would be a no-op, not a fault
            self._log_info(f"Skipping {self.fault_name} for decoder-only model (already causal)")
            self.is_injected = False
            return

        # Store original forward method
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        # Create faulty forward that applies causal masking
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with causal masking (wrong for encoder)."""
            bs, q_length, dim = query.size()
            k_length = key.size(1)

            # FAULTY: Create causal mask - WRONG for bidirectional encoder!
            causal_mask = torch.tril(torch.ones(q_length, k_length, device=query.device, dtype=torch.bool))
            causal_mask = causal_mask.view(1, 1, q_length, k_length)

            # Handle BERT-style attention
            if hasattr(attention, 'self'):
                self_attn = attention.self
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(self_attn.attention_head_size)
                scores = scores.masked_fill(~causal_mask, float('-inf'))

                if mask is not None:
                    scores = scores + mask

                weights = F.softmax(scores, dim=-1)
                weights = self_attn.dropout(weights)

                if head_mask is not None:
                    weights = weights * head_mask

                context = torch.matmul(weights, v)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_shape = context.size()[:-2] + (self_attn.all_head_size,)
                context = context.view(new_shape)
                context = attention.output.dense(context)
                context = attention.output.dropout(context)
                context = attention.output.LayerNorm(context + query)

                # Always return tuple (context, weights_or_none) for consistency
                return (context, weights if output_attentions else None)

            # Handle DistilBERT-style attention
            elif hasattr(attention, 'q_lin') and hasattr(attention, 'k_lin'):
                dim_per_head = (attention.dim or dim) // max(attention.n_heads or 1, 1)

                def shape(x):
                    return x.view(bs, -1, attention.n_heads, dim_per_head).transpose(1, 2)

                def unshape(x):
                    return x.transpose(1, 2).contiguous().view(bs, -1, attention.n_heads * dim_per_head)

                q = shape(attention.q_lin(query))
                k = shape(attention.k_lin(key))
                v = shape(attention.v_lin(value))

                scores = torch.matmul(q, k.transpose(2, 3))
                scores = scores / math.sqrt(max(dim_per_head, 1))
                scores = scores.masked_fill(~causal_mask, float('-inf'))

                mask_prepared = _prepare_mask(mask, bs, attention.n_heads, q_length, k_length, scores.dtype)
                if mask_prepared is not None:
                    scores = scores + mask_prepared

                weights = F.softmax(scores, dim=-1)

                if attention.training and attention.dropout is not None:
                    weights = attention.dropout(weights)

                weights = _apply_head_mask(weights, head_mask)

                context = torch.matmul(weights, v)
                context = unshape(context)
                context = attention.out_lin(context)

                # Always return tuple (context, weights_or_none) for consistency
                return (context, weights if output_attentions else None)

            # Handle GPT-2-style attention
            elif hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj'):
                embed_dim = getattr(attention, 'embed_dim', dim)
                num_heads = getattr(attention, 'num_heads', getattr(attention, 'n_heads', 1))
                head_dim = embed_dim // num_heads

                def split_heads(tensor):
                    new_shape = tensor.size()[:-1] + (num_heads, head_dim)
                    tensor = tensor.view(new_shape)
                    return tensor.permute(0, 2, 1, 3)

                def merge_heads(tensor):
                    tensor = tensor.permute(0, 2, 1, 3).contiguous()
                    new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
                    return tensor.view(new_shape)

                qkv = attention.c_attn(query)
                q, k, v = qkv.split(embed_dim, dim=2)
                q = split_heads(q)
                k = split_heads(k)
                v = split_heads(v)

                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(head_dim)
                scores = scores.masked_fill(~causal_mask, float('-inf'))

                if mask is not None:
                    scores = scores + mask

                weights = F.softmax(scores, dim=-1)

                if hasattr(attention, 'attn_dropout') and attention.attn_dropout is not None:
                    weights = attention.attn_dropout(weights)

                if head_mask is not None:
                    weights = weights * head_mask

                context = torch.matmul(weights, v)
                context = merge_heads(context)
                context = attention.c_proj(context)

                if hasattr(attention, 'resid_dropout') and attention.resid_dropout is not None:
                    context = attention.resid_dropout(context)

                # Always return tuple (context, weights_or_none) for consistency
                return (context, weights if output_attentions else None)

            else:
                raise ValueError(f"Unsupported attention architecture: {type(attention)}")

        # Replace forward method
        if hasattr(attention, 'self'):
            attention.self.forward = self._build_attention_wrapper(faulty_forward)
        else:
            attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
            if hasattr(attention, 'self'):
                attention.self.forward = self._backup_forward
            else:
                attention.forward = self._backup_forward

        self.original_forward = None
        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


# Registry of all variant faults
VARIANT_FAULTS = {
    'wrong_variant': WrongVariantFault,
    'causal_in_noncausal': CausalInNonCausalFault,
}


def create_variant_fault(fault_type: str, model: nn.Module, layer_idx: int = 2, **kwargs) -> AttentionFault:
    """
    Factory function to create variant faults.

    Args:
        fault_type: Type of fault ('wrong_variant', 'causal_in_noncausal')
        model: Model to inject fault into
        layer_idx: Target layer index
        **kwargs: Additional fault-specific parameters

    Returns:
        Initialized fault injector

    Raises:
        ValueError: If fault_type is unknown
    """
    if fault_type not in VARIANT_FAULTS:
        raise ValueError(
            f"Unknown fault type: {fault_type}. "
            f"Valid types: {list(VARIANT_FAULTS.keys())}"
        )

    fault_class = VARIANT_FAULTS[fault_type]
    return fault_class(model, layer_idx)
