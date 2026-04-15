"""
Masking Faults (E1 Category)

Implements three types of attention masking faults:
- E1.1: Zero Mask (drop padding mask)
- E1.2: Inverted Mask (invert mask values)
- E1.3: Wrong Mask Broadcast (cross-example leakage)

All faults target the attention mask mechanism in transformer layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from src.faults.base_fault import AttentionFault


class ZeroMaskFault(AttentionFault):
    """
    E1.1: Drop Padding Mask

    Removes the attention mask, causing tokens to attend to PAD positions.
    This simulates a bug where the padding mask is not applied.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize zero mask fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="zero_mask",
            description="Attention mask set to None, causing attention to PAD tokens"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the zero mask fault."""
        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        # Get the normalized forward function using the BACKUP (unwrapped) version
        original_forward_fn = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that drops mask
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with mask set to all zeros (allow all attention)."""
            # Create a zero mask (all positions allowed) instead of None
            if mask is not None:
                # Create all-zero mask with same shape as original
                zero_mask = torch.zeros_like(mask)
                mask = zero_mask

            # Call the original forward function directly (not through wrapper)
            return original_forward_fn(
                query, key, value,
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
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


class InvertedMaskFault(AttentionFault):
    """
    E1.2: Inverted Mask

    Inverts the attention mask values, causing wrong tokens to be masked/unmasked.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize inverted mask fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="inverted_mask",
            description="Attention mask values inverted (0↔1 swap)"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the inverted mask fault."""
        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        # Get the normalized forward function using the BACKUP (unwrapped) version
        original_forward_fn = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that inverts mask
        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with inverted mask."""
            # Invert mask if provided
            if mask is not None:
                # Convert to float if it's integer type to handle -inf
                if mask.dtype in [torch.int32, torch.int64, torch.long]:
                    mask = mask.float()

                # CRITICAL FIX (Bug #16): For decoder models (GPT-2/DistilGPT-2),
                # preserve causal structure while inverting padding behavior
                # Check if this is a 4D causal mask (seq_len x seq_len)
                if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                    # This is a decoder model with causal attention mask
                    seq_len = mask.shape[-1]

                    # Invert the mask values (swap 0 and -inf)
                    inverted_mask = torch.where(
                        torch.abs(mask) < 0.5,  # Detect unmasked (value near 0)
                        torch.full_like(mask, float('-inf')),
                        torch.zeros_like(mask)
                    )

                    # Restore causal structure: force upper triangle to remain -inf
                    # This preserves the causal constraint (no future attention)
                    upper_tri_indices = torch.triu_indices(seq_len, seq_len, offset=1, device=mask.device)
                    inverted_mask[..., upper_tri_indices[0], upper_tri_indices[1]] = float('-inf')

                    # NaN SAFETY FIX: Ensure diagonal is always unmasked to prevent NaN
                    # Simply set diagonal to 0 (allowed) - this is safe for causal attention
                    for i in range(seq_len):
                        inverted_mask[..., i, i] = 0.0

                    mask = inverted_mask
                else:
                    # For encoder models or 1D/2D masks (padding only), invert normally
                    # Additive mask: 0 for allowed, -inf for masked
                    # Invert by swapping 0 and -inf
                    inverted_mask = torch.where(
                        mask == 0,
                        torch.tensor(float('-inf'), dtype=mask.dtype, device=mask.device),
                        torch.zeros_like(mask)
                    )

                    # NaN SAFETY FIX: Ensure at least one position is unmasked
                    # Check if all positions are -inf (fully masked batch/sequence)
                    is_all_masked = (inverted_mask == float('-inf')).all(dim=-1, keepdim=True)
                    if is_all_masked.any():
                        # Unmask the first position for fully-masked sequences
                        inverted_mask[..., 0] = torch.where(
                            is_all_masked.squeeze(-1),
                            torch.zeros_like(inverted_mask[..., 0]),
                            inverted_mask[..., 0]
                        )

                    mask = inverted_mask

            # Call the original forward function directly (not through wrapper)
            return original_forward_fn(
                query, key, value,
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
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


class WrongMaskBroadcastFault(AttentionFault):
    """
    E1.3: Wrong Mask Broadcast (Cross-Example Leakage)

    Creates a uniform mask across batch, causing attention to leak between examples.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        """
        Initialize wrong mask broadcast fault.

        Args:
            model: The model to inject fault into
            layer_idx: Target layer (default: 2 for early layer)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="wrong_mask_broadcast",
            description="Mask broadcast incorrectly, causing cross-example leakage"
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the wrong mask broadcast fault."""
        if self.is_injected:
            return

        # Get target layer's attention module
        self.target_layer = self.get_layer()
        attention = self.get_attention_module()

        # Store original forward method BEFORE any wrapping
        if hasattr(attention, 'self'):
            self._backup_forward = attention.self.forward
        else:
            self._backup_forward = attention.forward

        # Get the normalized forward function using the BACKUP (unwrapped) version
        original_forward_fn = self._get_attention_forward(attention, forward_fn=self._backup_forward)

        # Create faulty forward that uses wrong broadcast

        def _normalize_mask(current_mask: torch.Tensor) -> torch.Tensor:
            """Convert mask to [batch, 1, 1, seq_len] additive form."""
            if current_mask is None:
                return current_mask
            if current_mask.dim() == 2:
                current_mask = current_mask.unsqueeze(1).unsqueeze(2)
            elif current_mask.dim() == 3:
                current_mask = current_mask.unsqueeze(1)
            return current_mask

        def faulty_forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
            """Forward pass with incorrectly broadcast mask."""
            if mask is not None:
                normalized = _normalize_mask(mask)
                if normalized is not None:
                    batch_size = normalized.size(0)
                    shared_mask = normalized[0:1].clone()

                    # NaN SAFETY FIX: Prevent over-masking that leads to all-inf scenarios
                    # Only broadcast if it won't create fully-masked sequences
                    expanded_masks = []
                    for i in range(batch_size):
                        example_mask = normalized[i:i+1]

                        # Count valid (non -inf) positions in both masks
                        num_valid_shared = (shared_mask != float('-inf')).sum()
                        num_valid_current = (example_mask != float('-inf')).sum()

                        # Use shared mask only if it has at least as many valid positions
                        # This prevents broadcasting a heavily-padded mask to longer sequences
                        if num_valid_shared >= num_valid_current or num_valid_current == 0:
                            expanded_masks.append(shared_mask)
                        else:
                            # Keep original mask to prevent over-masking
                            expanded_masks.append(example_mask)

                    mask = torch.cat(expanded_masks, dim=0)

            # Call the original forward function directly (not through wrapper)
            return original_forward_fn(
                query, key, value,
                mask=mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )

        # Replace forward method with signature-compatible wrapper
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


class BroadcastMaskFault(WrongMaskBroadcastFault):
    """Backward-compatible alias expected by static verification."""

    def __init__(self, model: nn.Module, layer_idx: int = 2):
        super().__init__(model=model, layer_idx=layer_idx)

    def inject(self) -> None:
        return super().inject()

    def remove(self) -> None:
        return super().restore()


# Registry of all masking faults
MASKING_FAULTS = {
    'zero_mask': ZeroMaskFault,
    'inverted_mask': InvertedMaskFault,
    'wrong_mask_broadcast': WrongMaskBroadcastFault,
}


def create_masking_fault(fault_type: str, model: nn.Module, layer_idx: int = 2) -> AttentionFault:
    """
    Factory function to create masking faults.

    Args:
        fault_type: Type of fault ('zero_mask', 'inverted_mask', 'wrong_mask_broadcast')
        model: Model to inject fault into
        layer_idx: Target layer index

    Returns:
        Initialized fault injector

    Raises:
        ValueError: If fault_type is unknown
    """
    if fault_type not in MASKING_FAULTS:
        raise ValueError(
            f"Unknown fault type: {fault_type}. "
            f"Valid types: {list(MASKING_FAULTS.keys())}"
        )

    return MASKING_FAULTS[fault_type](model, layer_idx)
