"""
QKV Faults (E2 Category)

Implements four types of Query-Key-Value projection faults:
- E2.1: Zero Query (Q projection set to zeros)
- E2.2: Zero Key (K projection set to zeros)
- E2.3: Zero Value (V projection set to zeros)
- E2.4: Swapped QK (Q and K projections swapped)

All faults target the QKV projection mechanism in transformer attention layers.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Callable, Tuple
from src.faults.base_fault import AttentionFault


class ZeroQueryFault(AttentionFault):
    """
    E2.1: Zero Query Projection

    Sets the query projection to zeros, preventing proper attention computation.
    This simulates a bug where Q = 0, making all attention scores zero after softmax.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize zero query fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="zero_query",
            description="Query projection set to zeros"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the zero query fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        self._backup_forward = attention.forward
        # Get normalized forward using the BACKUP (unwrapped) version
        self.original_forward = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that zeros Q
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with query set to zeros."""
            # Zero out the query
            query_zeroed = torch.zeros_like(query)

            # Call original forward with zeroed query
            return self.original_forward(
                query_zeroed, key, value,
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
        attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
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

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class ZeroKeyFault(AttentionFault):
    """
    E2.2: Zero Key Projection

    Sets the key projection to zeros, preventing proper attention computation.
    This simulates a bug where K = 0, making attention uniform after softmax.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize zero key fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="zero_key",
            description="Key projection set to zeros"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the zero key fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        self._backup_forward = attention.forward
        # Get normalized forward using the BACKUP (unwrapped) version
        self.original_forward = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that zeros K
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with key set to zeros."""
            # Zero out the key
            key_zeroed = torch.zeros_like(key)

            # Call original forward with zeroed key
            return self.original_forward(
                query, key_zeroed, value,
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
        attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
            attention.forward = self._backup_forward

        self.original_forward = None
        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class ZeroValueFault(AttentionFault):
    """
    E2.3: Zero Value Projection

    Sets the value projection to zeros, preventing information flow.
    This simulates a bug where V = 0, making output always zero.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize zero value fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="zero_value",
            description="Value projection set to zeros"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the zero value fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        self._backup_forward = attention.forward
        # Get normalized forward using the BACKUP (unwrapped) version
        self.original_forward = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that zeros V
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with value set to zeros."""
            # Zero out the value
            value_zeroed = torch.zeros_like(value)

            # Call original forward with zeroed value
            return self.original_forward(
                query, key, value_zeroed,
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
        attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
            attention.forward = self._backup_forward

        self.original_forward = None
        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class SwappedQKFault(AttentionFault):
    """
    E2.4: Swapped Query and Key Projections

    Swaps Q and K projections, breaking the attention mechanism's asymmetry.
    This simulates a bug where Q and K are accidentally swapped.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize swapped QK fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="swapped_qk",
            description="Query and Key projections swapped"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the swapped QK fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        self._backup_forward = attention.forward
        # Get normalized forward using the BACKUP (unwrapped) version
        self.original_forward = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that swaps Q and K
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with query and key swapped."""
            # Swap query and key
            return self.original_forward(
                key, query, value,  # Q and K swapped!
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
        attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = self.get_attention_module()
            attention.forward = self._backup_forward

        self.original_forward = None
        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class TieHeadsFault(AttentionFault):
    """
    E2.2: Tie All Heads to Same Projection

    Forces all attention heads to use the same projection weights.
    This simulates a bug where head diversity is lost.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="tie_heads",
            description="All heads tied to same projection"
        )
        self.original_weights = {}

    def inject(self) -> None:
        """Inject the tie heads fault."""
        if self.is_injected:
            return

        self.target_layer = self.get_layer()
        attention = self.get_attention_module()
        projections = self._get_qkv_projections(attention)
        q_proj = projections["q_proj"]
        k_proj = projections["k_proj"]
        v_proj = projections["v_proj"]

        # Store original weights
        self.original_weights['q'] = q_proj.weight.data.clone()
        self.original_weights['k'] = k_proj.weight.data.clone()
        self.original_weights['v'] = v_proj.weight.data.clone()

        # Get head dimension
        num_heads = projections["num_heads"] or getattr(getattr(self.model, "config", None), "num_attention_heads", 1)
        head_dim = q_proj.weight.size(0) // max(1, num_heads)

        # Tie all heads to head 0's weights
        for h in range(1, num_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            q_proj.weight.data[start_idx:end_idx] = q_proj.weight.data[:head_dim].clone()
            k_proj.weight.data[start_idx:end_idx] = k_proj.weight.data[:head_dim].clone()
            v_proj.weight.data[start_idx:end_idx] = v_proj.weight.data[:head_dim].clone()

        self.is_injected = True
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original weights."""
        if not self.is_injected:
            return

        attention = self.get_attention_module()
        projections = self._get_qkv_projections(attention)
        projections["q_proj"].weight.data.copy_(self.original_weights['q'])
        projections["k_proj"].weight.data.copy_(self.original_weights['k'])
        projections["v_proj"].weight.data.copy_(self.original_weights['v'])

        self.original_weights.clear()
        self.is_injected = False
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class WrongHeadDimFault(AttentionFault):
    """
    E2.3: Wrong Head Dimension

    Uses incorrect head dimension calculation.
    This simulates dimension mismatch bugs.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="wrong_head_dim",
            description="Wrong head dimension used"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the wrong head dimension fault."""
        if self.is_injected:
            return

        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward
        self._backup_forward = attention.forward
        projections = self._get_qkv_projections(attention)
        q_proj = projections["q_proj"]
        k_proj = projections["k_proj"]
        v_proj = projections["v_proj"]
        out_proj = projections["out_proj"]

        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward with wrong head dimension."""
            bs, q_length, dim = query.size()
            k_length = key.size(1)

            # Get actual dimensions
            actual_num_heads = projections["num_heads"] or getattr(attention, "n_heads", 1)
            actual_head_dim = (projections["dim"] or dim) // max(actual_num_heads, 1)

            # Use slightly wrong dimension - use num_heads - 1 to reduce effective capacity
            # This simulates the bug of using wrong head count
            wrong_num_heads = max(actual_num_heads - 1, 1)
            # Keep head dimension same but reduce number of heads
            head_dim = actual_head_dim

            def prepare_mask(attn_mask):
                if attn_mask is None:
                    return None
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask[:, None, None, :]
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, None, :, :]
                attn_mask = attn_mask.to(dtype=query.dtype)
                if attn_mask.dim() == 4 and attn_mask.size(1) == 1:
                    attn_mask = attn_mask.expand(bs, wrong_num_heads, q_length, k_length)
                return attn_mask

            def apply_head_mask(weights, mask):
                if mask is None:
                    return weights
                if mask.dim() == 1:
                    mask = mask.view(1, -1, 1, 1)
                elif mask.dim() == 2:
                    mask = mask.view(mask.size(0), mask.size(1), 1, 1)
                return weights * mask

            # Apply Q,K,V projections
            q_hidden = q_proj(query)  # (bs, q_length, full_dim)
            k_hidden = k_proj(key)    # (bs, k_length, full_dim)
            v_hidden = v_proj(value)  # (bs, k_length, full_dim)

            # FAULT: Use wrong number of heads (fewer than actual)
            # Truncate to wrong_num_heads * head_dim
            truncate_dim = wrong_num_heads * head_dim
            q_trunc = q_hidden[:, :, :truncate_dim]
            k_trunc = k_hidden[:, :, :truncate_dim]
            v_trunc = v_hidden[:, :, :truncate_dim]

            # Reshape with wrong number of heads
            q = q_trunc.view(bs, q_length, wrong_num_heads, head_dim).transpose(1, 2)
            k = k_trunc.view(bs, k_length, wrong_num_heads, head_dim).transpose(1, 2)
            v = v_trunc.view(bs, k_length, wrong_num_heads, head_dim).transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / math.sqrt(max(head_dim, 1))

            # Apply mask
            mask_prepared = prepare_mask(mask)
            if mask_prepared is not None:
                scores = scores + mask_prepared

            # Softmax
            weights = torch.nn.functional.softmax(scores, dim=-1)

            # Apply dropout (GPT-2 uses attn_dropout, BERT uses dropout)
            dropout_layer = getattr(attention, 'dropout', None) or getattr(attention, 'attn_dropout', None)
            if attention.training and dropout_layer is not None:
                weights = dropout_layer(weights)

            # Apply head mask
            weights = apply_head_mask(weights, head_mask)

            # Apply to values
            context = torch.matmul(weights, v)  # (bs, wrong_num_heads, q_length, head_dim)
            context = context.transpose(1, 2).contiguous()  # (bs, q_length, wrong_num_heads, head_dim)

            # Flatten
            context = context.view(bs, q_length, wrong_num_heads * head_dim)

            # Pad back to expected dimension (since we used fewer heads)
            expected_dim = projections["dim"] or dim
            current_dim = context.size(-1)
            if current_dim < expected_dim:
                padding_size = expected_dim - current_dim
                padding = torch.zeros(bs, q_length, padding_size, dtype=context.dtype, device=context.device)
                context = torch.cat([context, padding], dim=-1)
            elif current_dim > expected_dim:
                # Truncate if somehow larger
                context = context[:, :, :expected_dim]

            # Output projection
            if out_proj is not None:
                context = out_proj(context)

            # Return format should match original attention output
            # For GPT-2: (hidden_states, present_key_values) when use_cache=True
            # For GPT-2: (hidden_states, present_key_values, attn_weights) when both use_cache and output_attentions
            # Since we don't have access to use_cache param, always return compatible format
            # The wrapper will handle it
            if output_attentions:
                # Return (context, weights) - wrapper will add None for present if needed
                return (context, weights)
            # Return just context - wrapper will ensure tuple format
            return (context,)

        attention.forward = self._build_attention_wrapper(faulty_forward)
        self.is_injected = True
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward."""
        if not self.is_injected:
            return

        attention = self.get_attention_module()
        if self._backup_forward is not None:
            attention.forward = self._backup_forward
        self.original_forward = None
        self._backup_forward = None
        self.is_injected = False
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")

    def remove(self) -> None:
        """Alias for restore to satisfy static checks."""
        self.restore()


class FreezeQKVFault(AttentionFault):
    """
    E2.4: Freeze QKV Parameters

    Freezes QKV projection parameters so they don't update during training.
    This simulates accidentally excluding parameters from optimization.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="freeze_qkv",
            description="QKV parameters frozen (no gradient)"
        )
        self.original_grad_states = {}

    def inject(self) -> None:
        """Inject the freeze QKV fault."""
        if self.is_injected:
            return

        self.target_layer = self.get_layer()
        attention = self.get_attention_module()
        projections = self._get_qkv_projections(attention)
        q_proj = projections["q_proj"]
        k_proj = projections["k_proj"]
        v_proj = projections["v_proj"]

        # Check if we're dealing with GPT-2 style fused c_attn (virtual projections)
        if hasattr(attention, 'c_attn'):
            # For GPT-2, freeze the underlying c_attn parameter instead of virtual slices
            c_attn = attention.c_attn
            self.original_grad_states['c_attn_weight'] = c_attn.weight.requires_grad
            c_attn.weight.requires_grad_(False)
            if c_attn.bias is not None:
                self.original_grad_states['c_attn_bias'] = c_attn.bias.requires_grad
                c_attn.bias.requires_grad_(False)
        else:
            # For DistilBERT/BERT style, freeze individual Q/K/V projections
            # Use parameter() method to get the actual parameter, not a property/slice
            if hasattr(q_proj, 'weight') and isinstance(q_proj.weight, torch.nn.Parameter):
                self.original_grad_states['q'] = q_proj.weight.requires_grad
                self.original_grad_states['k'] = k_proj.weight.requires_grad
                self.original_grad_states['v'] = v_proj.weight.requires_grad
                q_proj.weight.requires_grad_(False)
                k_proj.weight.requires_grad_(False)
                v_proj.weight.requires_grad_(False)

                if q_proj.bias is not None:
                    self.original_grad_states['q_bias'] = q_proj.bias.requires_grad
                    self.original_grad_states['k_bias'] = k_proj.bias.requires_grad
                    self.original_grad_states['v_bias'] = v_proj.bias.requires_grad
                    q_proj.bias.requires_grad_(False)
                    k_proj.bias.requires_grad_(False)
                    v_proj.bias.requires_grad_(False)

        self.is_injected = True
        self._log_info(f"Injected {self.fault_name} into layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original grad states."""
        if not self.is_injected:
            return

        attention = self.get_attention_module()

        # Check if we froze c_attn (GPT-2 style) or individual projections
        if 'c_attn_weight' in self.original_grad_states:
            # Restore GPT-2 style c_attn
            c_attn = attention.c_attn
            c_attn.weight.requires_grad_(self.original_grad_states['c_attn_weight'])
            if 'c_attn_bias' in self.original_grad_states and c_attn.bias is not None:
                c_attn.bias.requires_grad_(self.original_grad_states['c_attn_bias'])
        else:
            # Restore DistilBERT/BERT style individual projections
            projections = self._get_qkv_projections(attention)
            if 'q' in self.original_grad_states:
                projections["q_proj"].weight.requires_grad_(self.original_grad_states['q'])
                projections["k_proj"].weight.requires_grad_(self.original_grad_states['k'])
                projections["v_proj"].weight.requires_grad_(self.original_grad_states['v'])

            if 'q_bias' in self.original_grad_states and projections["q_proj"].bias is not None:
                projections["q_proj"].bias.requires_grad_(self.original_grad_states['q_bias'])
                projections["k_proj"].bias.requires_grad_(self.original_grad_states['k_bias'])
                projections["v_proj"].bias.requires_grad_(self.original_grad_states['v_bias'])

        self.original_grad_states.clear()
        self.is_injected = False
        self._log_info(f"Restored layer {self.layer_idx} from {self.fault_name}")


# Registry of all QKV faults
QKV_FAULTS = {
    'zero_query': ZeroQueryFault,
    'zero_key': ZeroKeyFault,
    'zero_value': ZeroValueFault,
    'swapped_qk': SwappedQKFault,
    'tie_heads': TieHeadsFault,
    'wrong_head_dim': WrongHeadDimFault,
    'freeze_qkv': FreezeQKVFault,
}


def create_qkv_fault(fault_type: str, model: nn.Module, layer_idx: int = 2) -> AttentionFault:
    """
    Factory function to create QKV faults.

    Args:
        fault_type: Type of fault ('zero_query', 'zero_key', 'zero_value', 'swapped_qk')
        model: Model to inject fault into
        layer_idx: Target layer index

    Returns:
        Initialized fault injector

    Raises:
        ValueError: If fault_type is unknown
    """
    if fault_type not in QKV_FAULTS:
        raise ValueError(
            f"Unknown fault type: {fault_type}. "
            f"Valid types: {list(QKV_FAULTS.keys())}"
        )

    return QKV_FAULTS[fault_type](model, layer_idx)
