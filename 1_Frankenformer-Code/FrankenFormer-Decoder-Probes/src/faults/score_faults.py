"""
Score Faults (E3 Category)

Implements four types of attention score computation faults:
- E3.1: Missing Scaling (attention scores not scaled by √d_k)
- E3.2: Wrong Scaling Factor (incorrect scaling factor applied)
- E3.3: Misplaced Dropout (dropout applied before softmax)
- E3.4: Unsafe Type Cast (precision loss in score computation)

All faults target the attention score computation mechanism.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Callable
from src.faults.base_fault import AttentionFault

def _prepare_mask(mask, batch_size: int, n_heads: int, q_length: int, k_length: int, dtype: torch.dtype):
    """Broadcast attention mask to (bs, n_heads, q_len, k_len) with correct dtype."""
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
    """Safely apply optional head mask with broadcasting."""
    if head_mask is None:
        return weights
    if head_mask.dim() == 1:
        head_mask = head_mask.view(1, -1, 1, 1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
    return weights * head_mask


class MissingScalingFault(AttentionFault):
    """
    E3.1: Missing Scaling

    Removes the scaling by √d_k from attention scores.
    This simulates a bug where scores = Q @ K^T without division by √d_k.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize missing scaling fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="missing_scaling",
            description="Attention scores not scaled by √d_k"
        )
        self._backup_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the missing scaling fault."""
        if self.is_injected:
            return

        self.target_layer = self.get_layer()
        attention = self.get_attention_module()
        # Backup the correct forward method
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        num_heads = getattr(attention, "n_heads", None)
        if num_heads is None and hasattr(attention, "self"):
            num_heads = getattr(attention.self, "num_attention_heads", 1)
        if not num_heads:
            num_heads = 1
        dim_attr = getattr(attention, "dim", None)
        if dim_attr is None and hasattr(attention, "self"):
            dim_attr = getattr(attention.self, "all_head_size", None)

        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            # Handle BERT-style attention (has self.query, self.key, self.value)
            # Check this FIRST because _ensure_attention_aliases creates fake q_lin on BERT
            if hasattr(attention, 'self'):
                self_attn = attention.self

                # Get Q, K, V using BERT's projections
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                # FAULT: skip division by sqrt(dim_per_head)
                scores = torch.matmul(q, k.transpose(-1, -2))

                # Apply mask
                if mask is not None:
                    scores = scores + mask

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = self_attn.dropout(weights)

                # Apply head mask if provided
                if head_mask is not None:
                    weights = weights * head_mask

                # Compute context
                context = torch.matmul(weights, v)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_shape = context.size()[:-2] + (self_attn.all_head_size,)
                context = context.view(new_shape)

                # Apply output projection
                context = attention.output.dense(context)
                context = attention.output.dropout(context)
                context = attention.output.LayerNorm(context + query)

                if output_attentions:
                    return (context, weights)
                return (context,)

            # Handle DistilBERT-style attention (has q_lin, k_lin, v_lin - all three required)
            elif hasattr(attention, 'q_lin') and hasattr(attention, 'k_lin') and hasattr(attention, 'v_lin') and not hasattr(attention, 'self'):
                bs, q_length, dim = query.size()
                k_length = key.size(1)
                dim_per_head = (dim_attr or dim) // max(num_heads, 1)

                def shape(x):
                    return x.view(bs, -1, num_heads, dim_per_head).transpose(1, 2)

                def unshape(x):
                    return x.transpose(1, 2).contiguous().view(bs, -1, num_heads * dim_per_head)

                q = shape(attention.q_lin(query))
                k = shape(attention.k_lin(key))
                v = shape(attention.v_lin(value))

                # FAULT: skip division by sqrt(dim_per_head)
                scores = torch.matmul(q, k.transpose(2, 3))

                mask_prepared = _prepare_mask(mask, bs, num_heads, q_length, k_length, scores.dtype)
                if mask_prepared is not None:
                    scores = scores + mask_prepared

                weights = torch.nn.functional.softmax(scores, dim=-1)

                if attention.training and attention.dropout is not None:
                    weights = attention.dropout(weights)

                weights = _apply_head_mask(weights, head_mask)

                context = torch.matmul(weights, v)
                context = unshape(context)
                context = attention.out_lin(context)

                if output_attentions:
                    return (context, weights)
                return (context,)

            # Handle GPT-2-style attention (has c_attn combined QKV projection)
            elif hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj'):
                bs, q_length, dim = query.size()
                embed_dim = getattr(attention, 'embed_dim', dim)
                num_heads_gpt = getattr(attention, 'num_heads', num_heads)
                head_dim = embed_dim // num_heads_gpt

                def split_heads(tensor):
                    """Split into heads: (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)"""
                    new_shape = tensor.size()[:-1] + (num_heads_gpt, head_dim)
                    tensor = tensor.view(new_shape)
                    return tensor.permute(0, 2, 1, 3)

                def merge_heads(tensor):
                    """Merge heads: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)"""
                    tensor = tensor.permute(0, 2, 1, 3).contiguous()
                    new_shape = tensor.size()[:-2] + (num_heads_gpt * head_dim,)
                    return tensor.view(new_shape)

                # Apply combined QKV projection
                qkv = attention.c_attn(query)
                # Split into Q, K, V
                q, k, v = qkv.split(embed_dim, dim=2)

                # Split into multiple heads
                q = split_heads(q)
                k = split_heads(k)
                v = split_heads(v)

                # FAULT: skip division by sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-1, -2))

                # Apply attention mask if provided
                if mask is not None:
                    scores = scores + mask

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)

                # Apply attention dropout
                if hasattr(attention, 'attn_dropout') and attention.attn_dropout is not None:
                    weights = attention.attn_dropout(weights)

                # Apply head mask if provided
                if head_mask is not None:
                    weights = weights * head_mask

                # Compute context
                context = torch.matmul(weights, v)
                context = merge_heads(context)

                # Apply output projection
                context = attention.c_proj(context)

                # Apply residual dropout if it exists
                if hasattr(attention, 'resid_dropout') and attention.resid_dropout is not None:
                    context = attention.resid_dropout(context)

                if output_attentions:
                    return (context, weights)
                return (context,)

            else:
                raise ValueError(f"Unsupported attention architecture: {type(attention)}")

        # For BERT models, wrap attention.self.forward; for DistilBERT, wrap attention.forward
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

        self._backup_forward = None
        self.is_injected = False
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class WrongScalingFactorFault(AttentionFault):
    """
    E3.2: Wrong Scaling Factor

    Uses incorrect scaling factor for attention scores.
    This simulates a bug where scores are divided by wrong value (e.g., d_k instead of √d_k).
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2, wrong_factor: float = 2.0):
        """
        Initialize wrong scaling factor fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
            wrong_factor: Multiplier for the wrong scaling (default: 2.0 for 2x wrong scale)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="wrong_scaling_factor",
            description=f"Attention scores scaled by wrong factor ({wrong_factor}x)"
        )
        self.wrong_factor = wrong_factor
        self._backup_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the wrong scaling factor fault."""
        if self.is_injected:
            return

        self.target_layer = self.get_layer()
        attention = self.get_attention_module()
        # Backup the correct forward method
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        num_heads = getattr(attention, "n_heads", None)
        if num_heads is None and hasattr(attention, "self"):
            num_heads = getattr(attention.self, "num_attention_heads", 1)
        if not num_heads:
            num_heads = 1
        dim_attr = getattr(attention, "dim", None)
        if dim_attr is None and hasattr(attention, "self"):
            dim_attr = getattr(attention.self, "all_head_size", None)

        wrong_factor = float(self.wrong_factor)

        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            # Handle BERT-style attention (has self.query, self.key, self.value)
            # Check this FIRST because _ensure_attention_aliases creates fake q_lin on BERT
            if hasattr(attention, 'self'):
                self_attn = attention.self

                # Get Q, K, V using BERT's projections
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                # Compute scores with correct scaling
                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(self_attn.attention_head_size)
                # FAULT: apply wrong scaling factor
                scores = scores * wrong_factor

                # Apply mask
                if mask is not None:
                    scores = scores + mask

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = self_attn.dropout(weights)

                # Apply head mask if provided
                if head_mask is not None:
                    weights = weights * head_mask

                # Compute context
                context = torch.matmul(weights, v)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_shape = context.size()[:-2] + (self_attn.all_head_size,)
                context = context.view(new_shape)

                # Apply output projection
                context = attention.output.dense(context)
                context = attention.output.dropout(context)
                context = attention.output.LayerNorm(context + query)

                if output_attentions:
                    return (context, weights)
                return (context,)

            # Handle DistilBERT-style attention (has q_lin, k_lin, v_lin - all three required)
            elif hasattr(attention, 'q_lin') and hasattr(attention, 'k_lin') and hasattr(attention, 'v_lin') and not hasattr(attention, 'self'):
                bs, q_length, dim = query.size()
                k_length = key.size(1)
                dim_per_head = (dim_attr or dim) // max(num_heads, 1)

                def shape(x):
                    return x.view(bs, -1, num_heads, dim_per_head).transpose(1, 2)

                def unshape(x):
                    return x.transpose(1, 2).contiguous().view(bs, -1, num_heads * dim_per_head)

                q = shape(attention.q_lin(query))
                k = shape(attention.k_lin(key))
                v = shape(attention.v_lin(value))

                scores = torch.matmul(q, k.transpose(2, 3))
                scores = scores / math.sqrt(dim_per_head)
                scores = scores * wrong_factor

                mask_prepared = _prepare_mask(mask, bs, num_heads, q_length, k_length, scores.dtype)
                if mask_prepared is not None:
                    scores = scores + mask_prepared

                weights = torch.nn.functional.softmax(scores, dim=-1)

                if attention.training and attention.dropout is not None:
                    weights = attention.dropout(weights)

                weights = _apply_head_mask(weights, head_mask)

                context = torch.matmul(weights, v)
                context = unshape(context)
                context = attention.out_lin(context)

                if output_attentions:
                    return (context, weights)
                return (context,)

            # Handle GPT-2-style attention (has c_attn combined QKV projection)
            elif hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj'):
                bs, q_length, dim = query.size()
                embed_dim = getattr(attention, 'embed_dim', dim)
                num_heads_gpt = getattr(attention, 'num_heads', num_heads)
                head_dim = embed_dim // num_heads_gpt

                def split_heads(tensor):
                    new_shape = tensor.size()[:-1] + (num_heads_gpt, head_dim)
                    tensor = tensor.view(new_shape)
                    return tensor.permute(0, 2, 1, 3)

                def merge_heads(tensor):
                    tensor = tensor.permute(0, 2, 1, 3).contiguous()
                    new_shape = tensor.size()[:-2] + (num_heads_gpt * head_dim,)
                    return tensor.view(new_shape)

                qkv = attention.c_attn(query)
                q, k, v = qkv.split(embed_dim, dim=2)
                q = split_heads(q)
                k = split_heads(k)
                v = split_heads(v)

                # Compute scores with correct scaling, then apply wrong factor
                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(head_dim)
                # FAULT: apply wrong scaling factor
                scores = scores * wrong_factor

                if mask is not None:
                    scores = scores + mask

                weights = torch.nn.functional.softmax(scores, dim=-1)

                if hasattr(attention, 'attn_dropout') and attention.attn_dropout is not None:
                    weights = attention.attn_dropout(weights)

                if head_mask is not None:
                    weights = weights * head_mask

                context = torch.matmul(weights, v)
                context = merge_heads(context)
                context = attention.c_proj(context)

                if hasattr(attention, 'resid_dropout') and attention.resid_dropout is not None:
                    context = attention.resid_dropout(context)

                if output_attentions:
                    return (context, weights)
                return (context,)

            else:
                raise ValueError(f"Unsupported attention architecture: {type(attention)}")

        # For BERT models, wrap attention.self.forward; for DistilBERT, wrap attention.forward
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

        self._backup_forward = None
        self.is_injected = False
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class MisplacedDropoutFault(AttentionFault):
    """
    E3.3: Misplaced Dropout

    Applies dropout before softmax instead of after.
    This simulates a bug where dropout is applied to raw scores instead of attention weights.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2, dropout_p: float = 0.1):
        """
        Initialize misplaced dropout fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
            dropout_p: Dropout probability (default: 0.1)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="misplaced_dropout",
            description=f"Dropout (p={dropout_p}) applied before softmax instead of after"
        )
        self.dropout_p = dropout_p
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the misplaced dropout fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method
        # Backup the correct forward method
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        dropout_p = self.dropout_p

        # Create faulty forward that applies dropout before softmax
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with dropout applied before softmax."""
            # Handle BERT-style attention (has self.query, self.key, self.value)
            # Check this FIRST because _ensure_attention_aliases creates fake q_lin on BERT
            if hasattr(attention, 'self'):
                self_attn = attention.self

                # Get Q, K, V using BERT's projections
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(self_attn.attention_head_size)

                # Apply mask
                if mask is not None:
                    scores = scores + mask

                # FAULTY: Apply dropout BEFORE softmax
                if attention.training:
                    scores = torch.nn.functional.dropout(scores, p=dropout_p, training=True)

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)

                # Apply head mask if provided
                if head_mask is not None:
                    weights = weights * head_mask

                # Compute context
                context = torch.matmul(weights, v)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_shape = context.size()[:-2] + (self_attn.all_head_size,)
                context = context.view(new_shape)

                # Apply output projection
                context = attention.output.dense(context)
                context = attention.output.dropout(context)
                context = attention.output.LayerNorm(context + query)

                if output_attentions:
                    return (context, weights)
                else:
                    return (context,)

            # Handle DistilBERT-style attention (has q_lin, k_lin, v_lin - all three required)
            elif hasattr(attention, 'q_lin') and hasattr(attention, 'k_lin') and hasattr(attention, 'v_lin') and not hasattr(attention, 'self'):
                bs, q_length, dim = query.size()
                k_length = key.size(1)

                # Reshape for multi-head attention
                dim_per_head = (attention.dim or dim) // max(attention.n_heads or 1, 1)

                def shape(x):
                    """Reshape to (bs, n_heads, seq_len, dim_per_head)"""
                    return x.view(bs, -1, attention.n_heads, dim_per_head).transpose(1, 2)

                def unshape(x):
                    """Reshape back to (bs, seq_len, dim)"""
                    return x.transpose(1, 2).contiguous().view(bs, -1, attention.n_heads * dim_per_head)

                # Project Q, K, V
                q = shape(attention.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
                k = shape(attention.k_lin(key))    # (bs, n_heads, k_length, dim_per_head)
                v = shape(attention.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
                scores = scores / math.sqrt(dim_per_head)

                # Apply mask if present
                mask_prepared = _prepare_mask(mask, bs, attention.n_heads, q_length, k_length, scores.dtype)
                if mask_prepared is not None:
                    scores = scores + mask_prepared

                # FAULTY: Apply dropout BEFORE softmax
                if attention.training:
                    scores = torch.nn.functional.dropout(scores, p=dropout_p, training=True)

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)

                # Apply head mask if present
                weights = _apply_head_mask(weights, head_mask)

                # Apply attention to values
                context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
                context = unshape(context)  # (bs, q_length, dim)
                context = attention.out_lin(context)

                if output_attentions:
                    return (context, weights)
                else:
                    return (context,)

            # Handle GPT-2-style attention (has c_attn combined QKV projection)
            elif hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj'):
                bs, q_length, dim = query.size()
                embed_dim = getattr(attention, 'embed_dim', dim)
                num_heads_gpt = getattr(attention, 'num_heads', attention.n_heads)
                head_dim = embed_dim // num_heads_gpt

                def split_heads(tensor):
                    new_shape = tensor.size()[:-1] + (num_heads_gpt, head_dim)
                    tensor = tensor.view(new_shape)
                    return tensor.permute(0, 2, 1, 3)

                def merge_heads(tensor):
                    tensor = tensor.permute(0, 2, 1, 3).contiguous()
                    new_shape = tensor.size()[:-2] + (num_heads_gpt * head_dim,)
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

                # FAULTY: Apply dropout BEFORE softmax
                if attention.training:
                    scores = torch.nn.functional.dropout(scores, p=dropout_p, training=True)

                weights = torch.nn.functional.softmax(scores, dim=-1)

                if head_mask is not None:
                    weights = weights * head_mask

                context = torch.matmul(weights, v)
                context = merge_heads(context)
                context = attention.c_proj(context)

                if hasattr(attention, 'resid_dropout') and attention.resid_dropout is not None:
                    context = attention.resid_dropout(context)

                if output_attentions:
                    return (context, weights)
                return (context,)

            else:
                raise ValueError(f"Unsupported attention architecture: {type(attention)}")

        # Replace forward method with signature-compatible wrapper
        # For BERT models, wrap attention.self.forward; for DistilBERT, wrap attention.forward
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

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class UnsafeTypeCastFault(AttentionFault):
    """
    E3.4: Unsafe Type Cast

    Casts attention scores to lower precision (float16) causing precision loss.
    This simulates a bug where mixed precision is improperly applied.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize unsafe type cast fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="unsafe_type_cast",
            description="Attention scores cast to float16 causing precision loss"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the unsafe type cast fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method
        # Backup the correct forward method
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        # Create faulty forward that casts scores to float16
        def _unsafe_cast_scores(scores, mask_to_apply=None):
            original_dtype = scores.dtype
            scores_fp16 = scores.to(torch.float16)
            # Stabilize after unsafe cast to avoid NaN/inf propagation.
            scores_fp16 = torch.nan_to_num(scores_fp16, nan=0.0, posinf=1e4, neginf=-1e4)
            if mask_to_apply is not None:
                mask_fp16 = mask_to_apply.to(scores_fp16.dtype)
                mask_fp16 = torch.nan_to_num(mask_fp16, nan=0.0, posinf=1e4, neginf=-1e4)
                mask_fp16 = torch.clamp(mask_fp16, min=-1e4, max=1e4)
                scores_fp16 = scores_fp16 + mask_fp16
            scores_fp16 = torch.nan_to_num(scores_fp16, nan=0.0, posinf=1e4, neginf=-1e4)
            scores_fp16 = torch.clamp(scores_fp16, min=-1e4, max=1e4)
            return scores_fp16.to(original_dtype)

        def _sanitize_weights(weights):
            if not torch.isfinite(weights).all():
                weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            denom = weights.sum(dim=-1, keepdim=True)
            if (denom <= 0).any() or (not torch.isfinite(denom).all()):
                weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
                denom = weights.sum(dim=-1, keepdim=True)
                weights = torch.where(denom > 0, weights / denom, torch.zeros_like(weights))
            return weights

        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with unsafe type casting of scores."""
            # Handle BERT-style attention (has self.query, self.key, self.value)
            # Check this FIRST because _ensure_attention_aliases creates fake q_lin on BERT
            if hasattr(attention, 'self'):
                self_attn = attention.self

                # Get Q, K, V using BERT's projections
                q = self_attn.transpose_for_scores(self_attn.query(query))
                k = self_attn.transpose_for_scores(self_attn.key(key))
                v = self_attn.transpose_for_scores(self_attn.value(value))

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(self_attn.attention_head_size)

                # FAULTY: Cast to float16 and back (precision loss)
                scores = _unsafe_cast_scores(scores, mask)

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = _sanitize_weights(weights)
                weights = self_attn.dropout(weights)
                weights = _sanitize_weights(weights)

                # Apply head mask if provided
                if head_mask is not None:
                    weights = weights * head_mask

                # Compute context
                context = torch.matmul(weights, v)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_shape = context.size()[:-2] + (self_attn.all_head_size,)
                context = context.view(new_shape)

                # Apply output projection
                context = attention.output.dense(context)
                context = attention.output.dropout(context)
                context = attention.output.LayerNorm(context + query)

                if output_attentions:
                    return (context, weights)
                else:
                    return (context,)

            # Handle DistilBERT-style attention (has q_lin, k_lin, v_lin - all three required)
            elif hasattr(attention, 'q_lin') and hasattr(attention, 'k_lin') and hasattr(attention, 'v_lin') and not hasattr(attention, 'self'):
                bs, q_length, dim = query.size()
                k_length = key.size(1)

                # Reshape for multi-head attention
                dim_per_head = (attention.dim or dim) // max(attention.n_heads or 1, 1)

                def shape(x):
                    """Reshape to (bs, n_heads, seq_len, dim_per_head)"""
                    return x.view(bs, -1, attention.n_heads, dim_per_head).transpose(1, 2)

                def unshape(x):
                    """Reshape back to (bs, seq_len, dim)"""
                    return x.transpose(1, 2).contiguous().view(bs, -1, attention.n_heads * dim_per_head)

                # Project Q, K, V
                q = shape(attention.q_lin(query))
                k = shape(attention.k_lin(key))
                v = shape(attention.v_lin(value))

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(2, 3))
                scores = scores / math.sqrt(dim_per_head)

                # FAULTY: Cast to float16 and back (precision loss)
                mask_prepared = _prepare_mask(mask, bs, attention.n_heads, q_length, k_length, scores.dtype)
                scores = _unsafe_cast_scores(scores, mask_prepared)

                # Apply softmax
                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = _sanitize_weights(weights)

                # Apply dropout if training
                if attention.training and attention.dropout is not None:
                    weights = attention.dropout(weights)
                    weights = _sanitize_weights(weights)

                # Apply head mask if present
                weights = _apply_head_mask(weights, head_mask)

                # Apply attention to values
                context = torch.matmul(weights, v)
                context = unshape(context)
                context = attention.out_lin(context)

                if output_attentions:
                    return (context, weights)
                else:
                    return (context,)

            # Handle GPT-2-style attention (has c_attn combined QKV projection)
            elif hasattr(attention, 'c_attn') and hasattr(attention, 'c_proj'):
                bs, q_length, dim = query.size()
                embed_dim = getattr(attention, 'embed_dim', dim)
                num_heads_gpt = getattr(attention, 'num_heads', attention.n_heads)
                head_dim = embed_dim // num_heads_gpt

                def split_heads(tensor):
                    new_shape = tensor.size()[:-1] + (num_heads_gpt, head_dim)
                    tensor = tensor.view(new_shape)
                    return tensor.permute(0, 2, 1, 3)

                def merge_heads(tensor):
                    tensor = tensor.permute(0, 2, 1, 3).contiguous()
                    new_shape = tensor.size()[:-2] + (num_heads_gpt * head_dim,)
                    return tensor.view(new_shape)

                qkv = attention.c_attn(query)
                q, k, v = qkv.split(embed_dim, dim=2)
                q = split_heads(q)
                k = split_heads(k)
                v = split_heads(v)

                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(head_dim)

                # FAULTY: Cast to float16 and back (precision loss)
                scores = _unsafe_cast_scores(scores, mask)

                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = _sanitize_weights(weights)

                if hasattr(attention, 'attn_dropout') and attention.attn_dropout is not None:
                    weights = attention.attn_dropout(weights)
                    weights = _sanitize_weights(weights)

                if head_mask is not None:
                    weights = weights * head_mask

                context = torch.matmul(weights, v)
                context = merge_heads(context)
                context = attention.c_proj(context)

                if hasattr(attention, 'resid_dropout') and attention.resid_dropout is not None:
                    context = attention.resid_dropout(context)

                if output_attentions:
                    return (context, weights)
                return (context,)

            else:
                raise ValueError(f"Unsupported attention architecture: {type(attention)}")

        # Replace forward method with signature-compatible wrapper
        # For BERT models, wrap attention.self.forward; for DistilBERT, wrap attention.forward
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


# Registry of all score faults
SCORE_FAULTS = {
    'missing_scaling': MissingScalingFault,
    'wrong_scaling_factor': WrongScalingFactorFault,
    'misplaced_dropout': MisplacedDropoutFault,
    'unsafe_type_cast': UnsafeTypeCastFault,
}


def create_score_fault(fault_type: str, model: nn.Module, layer_idx: int = 2, **kwargs) -> AttentionFault:
    """
    Factory function to create score faults.

    Args:
        fault_type: Type of fault ('missing_scaling', 'wrong_scaling_factor', 'misplaced_dropout', 'unsafe_type_cast')
        model: Model to inject fault into
        layer_idx: Target layer index
        **kwargs: Additional fault-specific parameters

    Returns:
        Initialized fault injector

    Raises:
        ValueError: If fault_type is unknown
    """
    if fault_type not in SCORE_FAULTS:
        raise ValueError(
            f"Unknown fault type: {fault_type}. "
            f"Valid types: {list(SCORE_FAULTS.keys())}"
        )

    fault_class = SCORE_FAULTS[fault_type]

    # Pass kwargs for faults that need them
    if fault_type == 'wrong_scaling_factor':
        return fault_class(model, layer_idx, wrong_factor=kwargs.get('wrong_factor', 2.0))
    elif fault_type == 'misplaced_dropout':
        return fault_class(model, layer_idx, dropout_p=kwargs.get('dropout_p', 0.1))
    else:
        return fault_class(model, layer_idx)
