"""
Kernel Faults (E5 Category)

Implements three types of kernel/integration faults:
- E5.1: Force Unoptimized Kernel (disable optimized attention implementations)
- E5.2: Wrong Layout Flag (incorrect is_causal flag)
- E5.3: Inconsistent Dropout (dropout mismatch between passes)

All faults target the kernel/backend integration in attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable
from src.faults.base_fault import AttentionFault

def _prepare_mask(mask, batch_size: int, n_heads: int, q_length: int, k_length: int, dtype: torch.dtype):
    """Broadcast attention mask to match attention score shape."""
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


class ForceUnoptimizedFault(AttentionFault):
    """
    E5.1: Force Unoptimized Kernel

    Forces use of unoptimized attention implementation by disabling CUDNN.
    This simulates a bug where optimized kernels are not used.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize force unoptimized fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="force_unoptimized",
            description="Forces unoptimized attention kernel (CUDNN disabled)"
        )
        self.original_forward: Optional[Callable] = None
        self.original_cudnn_enabled: Optional[bool] = None

    def inject(self) -> None:
        """Inject the force unoptimized fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original CUDNN state
        self.original_cudnn_enabled = torch.backends.cudnn.enabled

        # Store original forward method BEFORE any wrapping
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward
        # Get normalized forward using the BACKUP (unwrapped) version
        self.original_forward = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that disables CUDNN
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with CUDNN disabled (unoptimized kernel)."""
            # Temporarily disable CUDNN
            original_state = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False

            try:
                # Call original forward with CUDNN disabled
                result = self.original_forward(
                    query, key, value,
                    mask=mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions
                )
            finally:
                # Restore CUDNN state
                torch.backends.cudnn.enabled = original_state

            return result

        # Replace forward method with signature-compatible wrapper
        if hasattr(attention, 'self'):
            attention.self.forward = self._build_attention_wrapper(faulty_forward)
        else:
            attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True
        self._set_kernel_fault_state('force_unoptimized', True)

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method and CUDNN state."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
            if hasattr(attention, 'self'):
                attention.self.forward = self._backup_forward
            else:
                attention.forward = self._backup_forward

        if self.original_cudnn_enabled is not None:
            torch.backends.cudnn.enabled = self.original_cudnn_enabled

        self.original_forward = None
        self._backup_forward = None
        self.original_cudnn_enabled = None
        self.is_injected = False
        self._set_kernel_fault_state('force_unoptimized', False)

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class WrongLayoutFault(AttentionFault):
    """
    E5.2: Wrong Layout Flag

    Uses is_causal=True for encoder attention (should be non-causal).
    This simulates a bug where causal masking is incorrectly applied.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize wrong layout fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="wrong_layout",
            description="Incorrectly uses is_causal=True for encoder attention"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the wrong layout fault."""
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

        # Create faulty forward that uses wrong causal flag
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with incorrect is_causal flag."""
            bs, q_length, dim = query.size()
            k_length = key.size(1)

            # Detect if this is a decoder-only model (GPT-2) vs encoder (BERT/DistilBERT)
            is_decoder_model = hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj')

            # FAULTY: Apply WRONG attention pattern based on model type
            # - For encoder (BERT/DistilBERT): Apply causal mask (wrong, should be bidirectional)
            # - For decoder (GPT-2): Remove causal constraint (wrong, should be causal)
            if is_decoder_model:
                # For decoder: WRONG layout = allow bidirectional attention (no causal mask)
                # This is a fault because decoder should only attend to past tokens
                faulty_mask = None  # No causal constraint = bidirectional attention
            else:
                # For encoder: WRONG layout = apply causal masking (prevents future attention)
                # This is a fault because encoder should be bidirectional
                causal_mask = torch.tril(torch.ones(q_length, k_length, device=query.device, dtype=torch.bool))
                causal_mask = causal_mask.view(1, 1, q_length, k_length)
                faulty_mask = causal_mask

            # Handle BERT-style attention
            if hasattr(attention, 'self'):
                self_attn = attention.self
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(self_attn.attention_head_size)
                if faulty_mask is not None:
                    scores = scores.masked_fill(~faulty_mask, float('-inf'))

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
                if faulty_mask is not None:
                    scores = scores.masked_fill(~faulty_mask, float('-inf'))

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
                # For GPT-2, faulty_mask is None (bidirectional attention = wrong for decoder)
                # Don't apply any causal masking - this allows attending to future tokens (FAULT!)
                if faulty_mask is not None:
                    scores = scores.masked_fill(~faulty_mask, float('-inf'))

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
        self._set_kernel_fault_state('wrong_layout', True)

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
        self._set_kernel_fault_state('wrong_layout', False)

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


class InconsistentDropoutFault(AttentionFault):
    """
    E5.3: Inconsistent Dropout

    Uses different dropout probabilities in different parts of attention.
    This simulates a bug where dropout configuration is inconsistent.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2, wrong_dropout_p: float = 0.5):
        """
        Initialize inconsistent dropout fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
            wrong_dropout_p: Wrong dropout probability to use (default: 0.5)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="inconsistent_dropout",
            description=f"Inconsistent dropout (p={wrong_dropout_p} instead of configured value)"
        )
        self.wrong_dropout_p = wrong_dropout_p
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the inconsistent dropout fault."""
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

        wrong_p = self.wrong_dropout_p

        # Create faulty forward with inconsistent dropout
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with inconsistent dropout probability."""
            bs, q_length, dim = query.size()
            k_length = key.size(1)

            # Handle BERT-style attention
            if hasattr(attention, 'self'):
                self_attn = attention.self
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(self_attn.attention_head_size)

                if mask is not None:
                    scores = scores + mask

                weights = F.softmax(scores, dim=-1)
                # FAULTY: Apply dropout with WRONG probability
                if attention.training:
                    weights = F.dropout(weights, p=wrong_p, training=True)

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

                mask_prepared = _prepare_mask(mask, bs, attention.n_heads, q_length, k_length, scores.dtype)
                if mask_prepared is not None:
                    scores = scores + mask_prepared

                weights = F.softmax(scores, dim=-1)

                # FAULTY: Apply dropout with WRONG probability
                if attention.training:
                    weights = F.dropout(weights, p=wrong_p, training=True)

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

                if mask is not None:
                    scores = scores + mask

                weights = F.softmax(scores, dim=-1)

                # FAULTY: Apply dropout with WRONG probability
                if attention.training:
                    weights = F.dropout(weights, p=wrong_p, training=True)

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
        self._set_kernel_fault_state('inconsistent_dropout', True)

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
        self._set_kernel_fault_state('inconsistent_dropout', False)

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


# Registry of all kernel faults
KERNEL_FAULTS = {
    'force_unoptimized': ForceUnoptimizedFault,
    'wrong_layout': WrongLayoutFault,
    'inconsistent_dropout': InconsistentDropoutFault,
}


def create_kernel_fault(fault_type: str, model: nn.Module, layer_idx: int = 2, **kwargs) -> AttentionFault:
    """
    Factory function to create kernel faults.

    Args:
        fault_type: Type of fault ('force_unoptimized', 'wrong_layout', 'inconsistent_dropout')
        model: Model to inject fault into
        layer_idx: Target layer index
        **kwargs: Additional fault-specific parameters

    Returns:
        Initialized fault injector

    Raises:
        ValueError: If fault_type is unknown
    """
    if fault_type not in KERNEL_FAULTS:
        raise ValueError(
            f"Unknown fault type: {fault_type}. "
            f"Valid types: {list(KERNEL_FAULTS.keys())}"
        )

    fault_class = KERNEL_FAULTS[fault_type]

    # Pass kwargs for faults that need them
    if fault_type == 'inconsistent_dropout':
        return fault_class(model, layer_idx, wrong_dropout_p=kwargs.get('wrong_dropout_p', 0.5))
    else:
        return fault_class(model, layer_idx)
