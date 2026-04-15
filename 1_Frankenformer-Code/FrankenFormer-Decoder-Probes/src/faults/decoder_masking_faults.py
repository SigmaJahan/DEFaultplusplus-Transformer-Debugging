"""
Decoder-Specific Masking Faults

Implements three types of causal attention masking faults for decoder models:
- Break Causal Mask: Allow attention to future positions
- Over Mask Valid Tokens: Mask out valid past tokens
- PAD Masking Error: Corrupt padding masks in batched sequences

All faults target the causal attention mask mechanism in decoder layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
from src.faults.base_fault import AttentionFault
from src.faults.attention_utils import (
    get_causal_mask,
    break_causal_mask,
    over_mask_valid_tokens,
    is_causal_mask,
    is_causal_architecture
)

def _get_attention_mask(kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    # Avoid boolean evaluation on tensors while preserving key priority.
    for key in ("attention_mask", "attn_mask", "mask"):
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
    return None


class BreakCausalMaskFault(AttentionFault):
    """
    Break Causal Mask Fault

    Allows decoder to attend to future positions, violating the causal constraint.
    This simulates a bug where the causal mask is incorrectly implemented.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        visibility_ratio: float = 0.5,
    ):
        """
        Initialize break causal mask fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Target layer index
            visibility_ratio: Fraction of future positions to make visible (0.0 to 1.0)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="break_causal_mask",
            description=f"Allow {visibility_ratio*100:.0f}% attention to future positions"
        )
        self.visibility_ratio = visibility_ratio
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the break causal mask fault."""
        self.target_layer = self.get_decoder_layer()

        # Get attention module from decoder layer
        attention = None
        if hasattr(self.target_layer, 'attn'):
            attention = self.target_layer.attn
        elif hasattr(self.target_layer, 'attention'):
            attention = self.target_layer.attention
        elif hasattr(self.target_layer, 'self_attn'):
            attention = self.target_layer.self_attn

        if attention is None:
            raise ValueError(f"Cannot find attention module in decoder layer {self.layer_idx}")

        # Store original forward method
        self._backup_forward = attention.forward

        # Create faulty forward that breaks causal mask
        visibility_ratio = self.visibility_ratio

        def faulty_forward(*args, **kwargs):
            """
            Forward pass with broken causal mask.

            CRITICAL FIX: For decoder architectures (like GPT-2), the causal mask is often
            internal, and attention_mask is just padding. We need to inject a broken causal
            mask or modify the internal causal mechanism.
            """
            # Get attention mask from kwargs or args
            # Common patterns: attention_mask, attn_mask, mask
            attention_mask = _get_attention_mask(kwargs)

            # CRITICAL FIX: Check if this is a decoder architecture AND has a mask
            # For GPT-2 style: attention_mask is padding-only, causal is internal
            # For explicit causal: attention_mask already contains causal structure
            is_decoder_arch = is_causal_architecture(self.model)

            if attention_mask is not None:
                if is_causal_mask(attention_mask):
                    # Explicit causal mask - break it directly
                    broken_mask = break_causal_mask(attention_mask, visibility_ratio)
                elif is_decoder_arch and attention_mask.dim() >= 2:
                    # Decoder architecture with padding-only mask
                    # Create a broken causal mask and combine with padding
                    seq_len = attention_mask.shape[-1]
                    device = attention_mask.device
                    dtype = attention_mask.dtype

                    # Generate full causal mask
                    full_causal = get_causal_mask(seq_len, device, dtype)

                    # Break the causal constraint
                    broken_causal = break_causal_mask(full_causal, visibility_ratio)

                    # Combine with padding mask
                    # Padding mask: 0 = pad (mask), 1 = valid
                    # Convert to additive: (1 - mask) * -inf
                    if attention_mask.dim() == 2:
                        # (batch, seq_len) -> (batch, 1, 1, seq_len) for broadcasting
                        mask_view = attention_mask.unsqueeze(1).unsqueeze(2)
                        padding_additive = torch.zeros_like(mask_view, dtype=dtype)
                    elif attention_mask.dim() == 3:
                        # (batch, seq_len, seq_len)
                        mask_view = attention_mask
                        padding_additive = torch.zeros_like(mask_view, dtype=dtype)
                    elif attention_mask.dim() == 4:
                        # (batch, heads, seq_len, seq_len)
                        mask_view = attention_mask
                        padding_additive = torch.zeros_like(mask_view, dtype=dtype)
                    else:
                        # Unsupported shape, skip
                        return self._backup_forward(*args, **kwargs)
                    # Avoid 0 * -inf -> NaN by using masked_fill
                    padding_additive = padding_additive.masked_fill(mask_view <= 0, float('-inf'))

                    # Combine: take minimum (more restrictive)
                    broken_mask = torch.minimum(broken_causal, padding_additive)
                else:
                    # Not a decoder or unsupported format
                    return self._backup_forward(*args, **kwargs)

                # Update kwargs
                if 'attention_mask' in kwargs:
                    kwargs['attention_mask'] = broken_mask
                elif 'attn_mask' in kwargs:
                    kwargs['attn_mask'] = broken_mask
                elif 'mask' in kwargs:
                    kwargs['mask'] = broken_mask

            return self._backup_forward(*args, **kwargs)

        attention.forward = faulty_forward
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into decoder layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            # Get attention module
            attention = None
            if hasattr(self.target_layer, 'attn'):
                attention = self.target_layer.attn
            elif hasattr(self.target_layer, 'attention'):
                attention = self.target_layer.attention
            elif hasattr(self.target_layer, 'self_attn'):
                attention = self.target_layer.self_attn

            if attention is not None:
                attention.forward = self._backup_forward

        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored decoder layer {self.layer_idx} from {self.fault_name}")


class OverMaskValidTokensFault(AttentionFault):
    """
    Over Mask Valid Tokens Fault

    Masks out some valid past tokens that should be visible in causal attention.
    This simulates a bug where the causal window is incorrectly truncated.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        mask_ratio: float = 0.2,
    ):
        """
        Initialize over mask valid tokens fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Target layer index
            mask_ratio: Fraction of valid past positions to mask out (0.0 to 1.0)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="over_mask_valid",
            description=f"Mask out {mask_ratio*100:.0f}% of valid past tokens"
        )
        self.mask_ratio = mask_ratio
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the over mask fault."""
        self.target_layer = self.get_decoder_layer()

        # Get attention module
        attention = None
        if hasattr(self.target_layer, 'attn'):
            attention = self.target_layer.attn
        elif hasattr(self.target_layer, 'attention'):
            attention = self.target_layer.attention
        elif hasattr(self.target_layer, 'self_attn'):
            attention = self.target_layer.self_attn

        if attention is None:
            raise ValueError(f"Cannot find attention module in decoder layer {self.layer_idx}")

        # Store original forward method
        self._backup_forward = attention.forward

        # Create faulty forward that over-masks valid tokens
        mask_ratio = self.mask_ratio

        def faulty_forward(*args, **kwargs):
            """
            Forward pass with over-masked valid tokens.

            CRITICAL FIX: Handle both explicit causal masks and decoder architectures
            with internal causal masks (like GPT-2).
            """
            attention_mask = _get_attention_mask(kwargs)

            is_decoder_arch = is_causal_architecture(self.model)

            if attention_mask is not None:
                if is_causal_mask(attention_mask):
                    # Explicit causal mask - over-mask directly
                    over_masked = over_mask_valid_tokens(attention_mask, mask_ratio)
                elif is_decoder_arch and attention_mask.dim() >= 2:
                    # Decoder with padding-only mask
                    seq_len = attention_mask.shape[-1]
                    device = attention_mask.device
                    dtype = attention_mask.dtype

                    # Generate full causal mask
                    full_causal = get_causal_mask(seq_len, device, dtype)

                    # Over-mask valid positions
                    over_masked_causal = over_mask_valid_tokens(full_causal, mask_ratio)

                    # Combine with padding mask
                    if attention_mask.dim() == 2:
                        mask_view = attention_mask.unsqueeze(1).unsqueeze(2)
                        padding_additive = torch.zeros_like(mask_view, dtype=dtype)
                    elif attention_mask.dim() == 3:
                        mask_view = attention_mask
                        padding_additive = torch.zeros_like(mask_view, dtype=dtype)
                    elif attention_mask.dim() == 4:
                        mask_view = attention_mask
                        padding_additive = torch.zeros_like(mask_view, dtype=dtype)
                    else:
                        return self._backup_forward(*args, **kwargs)
                    padding_additive = padding_additive.masked_fill(mask_view <= 0, float('-inf'))

                    # Combine: take minimum (more restrictive)
                    over_masked = torch.minimum(over_masked_causal, padding_additive)
                else:
                    return self._backup_forward(*args, **kwargs)

                # Update kwargs
                if 'attention_mask' in kwargs:
                    kwargs['attention_mask'] = over_masked
                elif 'attn_mask' in kwargs:
                    kwargs['attn_mask'] = over_masked
                elif 'mask' in kwargs:
                    kwargs['mask'] = over_masked

            return self._backup_forward(*args, **kwargs)

        attention.forward = faulty_forward
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into decoder layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = None
            if hasattr(self.target_layer, 'attn'):
                attention = self.target_layer.attn
            elif hasattr(self.target_layer, 'attention'):
                attention = self.target_layer.attention
            elif hasattr(self.target_layer, 'self_attn'):
                attention = self.target_layer.self_attn

            if attention is not None:
                attention.forward = self._backup_forward

        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored decoder layer {self.layer_idx} from {self.fault_name}")


class PadMaskingErrorFault(AttentionFault):
    """
    PAD Masking Error Fault

    Corrupts padding mask handling in batched sequences, causing attention
    to PAD tokens or exclusion of valid tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        error_type: str = "allow_pad_attention",
        batch_corruption: float = 0.5,
    ):
        """
        Initialize PAD masking error fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Target layer index
            error_type: Type of PAD error ("allow_pad_attention", "exclude_valid", "mixed")
            batch_corruption: Fraction of batch to corrupt (0.0 to 1.0)
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="pad_masking_error",
            description=f"PAD mask error ({error_type}) in {batch_corruption*100:.0f}% of batch"
        )
        self.error_type = error_type
        self.batch_corruption = batch_corruption
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        """Inject the PAD masking error fault."""
        self.target_layer = self.get_decoder_layer()

        # Get attention module
        attention = None
        if hasattr(self.target_layer, 'attn'):
            attention = self.target_layer.attn
        elif hasattr(self.target_layer, 'attention'):
            attention = self.target_layer.attention
        elif hasattr(self.target_layer, 'self_attn'):
            attention = self.target_layer.self_attn

        if attention is None:
            raise ValueError(f"Cannot find attention module in decoder layer {self.layer_idx}")

        # Store original forward method
        self._backup_forward = attention.forward

        # Create faulty forward that corrupts PAD masking
        error_type = self.error_type
        batch_corruption = self.batch_corruption

        def faulty_forward(*args, **kwargs):
            """Forward pass with corrupted PAD masking."""
            attention_mask = _get_attention_mask(kwargs)

            if attention_mask is not None:
                batch_size = attention_mask.shape[0] if attention_mask.dim() >= 2 else 1

                # Determine which batch items to corrupt
                num_to_corrupt = max(1, int(batch_size * batch_corruption))

                if num_to_corrupt > 0 and batch_size > 1:
                    # Random batch indices to corrupt
                    corrupt_indices = torch.randperm(batch_size, device=attention_mask.device)[:num_to_corrupt]

                    # Clone mask
                    corrupted_mask = attention_mask.clone()

                    for idx in corrupt_indices:
                        if error_type == "allow_pad_attention":
                            # Allow attention to all positions (including PAD)
                            # This simulates ignoring the padding mask
                            item_mask = corrupted_mask[idx]
                            if item_mask.abs().max() < 10:
                                # Binary mask: 1 = valid
                                corrupted_mask[idx] = 1.0
                            else:
                                # Additive mask: 0 means not masked
                                corrupted_mask[idx] = 0.0

                        elif error_type == "exclude_valid":
                            # Randomly exclude some valid (non-PAD) tokens
                            # CRITICAL FIX: Handle 4D masks correctly
                            # For 2D mask: (seq_len,) or (seq_len, seq_len)
                            # For 3D mask: (batch, seq_len, seq_len)
                            # For 4D mask: (batch, heads, seq_len, seq_len)

                            item_mask = corrupted_mask[idx]

                            # Identify PAD positions (highly negative values)
                            # Also handle binary padding masks (0 = pad, 1 = valid)
                            if item_mask.abs().max() < 10:
                                # Binary mask: 0 = pad, 1 = valid
                                valid_positions = item_mask > 0.5
                            else:
                                # Additive mask: -inf or large negative = pad
                                pad_positions = item_mask < -1e4
                                valid_positions = ~pad_positions

                            # CRITICAL FIX: Properly handle different mask dimensions
                            # For 4D mask [heads, seq, seq], we want to mask positions in the last dimension
                            # For 2D mask [seq, seq], same logic
                            if valid_positions.dim() >= 2:
                                # Find valid positions in the last dimension
                                # Sum over all but the last dimension to find which positions are valid
                                valid_in_last_dim = valid_positions.any(dim=tuple(range(valid_positions.dim() - 1)))
                                valid_indices = torch.where(valid_in_last_dim)[0]
                            else:
                                # 1D mask case
                                valid_indices = torch.where(valid_positions)[0]

                            # Mask out 30% of valid positions
                            if valid_indices.numel() > 0:
                                num_to_mask = max(1, valid_indices.numel() // 3)
                                mask_indices = valid_indices[torch.randperm(valid_indices.numel(), device=attention_mask.device)[:num_to_mask]]
                                # Apply masking to all dimensions except the one we're indexing
                                corrupted_mask[idx, ..., mask_indices] = float('-inf')

                        elif error_type == "mixed":
                            # Mix of both errors
                            if torch.rand(1).item() > 0.5:
                                corrupted_mask[idx] = 0.0
                            else:
                                # CRITICAL FIX: Same fix as "exclude_valid" above
                                item_mask = corrupted_mask[idx]

                                # Identify PAD positions (highly negative values)
                                # Also handle binary padding masks (0 = pad, 1 = valid)
                                if item_mask.abs().max() < 10:
                                    # Binary mask: 0 = pad, 1 = valid
                                    valid_positions = item_mask > 0.5
                                else:
                                    # Additive mask: -inf or large negative = pad
                                    pad_positions = item_mask < -1e4
                                    valid_positions = ~pad_positions

                                # CRITICAL FIX: Properly handle different mask dimensions
                                if valid_positions.dim() >= 2:
                                    valid_in_last_dim = valid_positions.any(dim=tuple(range(valid_positions.dim() - 1)))
                                    valid_indices = torch.where(valid_in_last_dim)[0]
                                else:
                                    valid_indices = torch.where(valid_positions)[0]

                                if valid_indices.numel() > 0:
                                    num_to_mask = max(1, valid_indices.numel() // 3)
                                    mask_indices = valid_indices[torch.randperm(valid_indices.numel(), device=attention_mask.device)[:num_to_mask]]
                                    corrupted_mask[idx, ..., mask_indices] = float('-inf')

                    # Update kwargs
                    if 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = corrupted_mask
                    elif 'attn_mask' in kwargs:
                        kwargs['attn_mask'] = corrupted_mask
                    elif 'mask' in kwargs:
                        kwargs['mask'] = corrupted_mask

            return self._backup_forward(*args, **kwargs)

        attention.forward = faulty_forward
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} into decoder layer {self.layer_idx}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            attention = None
            if hasattr(self.target_layer, 'attn'):
                attention = self.target_layer.attn
            elif hasattr(self.target_layer, 'attention'):
                attention = self.target_layer.attention
            elif hasattr(self.target_layer, 'self_attn'):
                attention = self.target_layer.self_attn

            if attention is not None:
                attention.forward = self._backup_forward

        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored decoder layer {self.layer_idx} from {self.fault_name}")


# Registry of decoder masking faults
DECODER_MASKING_FAULTS = {
    "break_causal_mask": BreakCausalMaskFault,
    "over_mask_valid": OverMaskValidTokensFault,
    "over_mask_valid_tokens": OverMaskValidTokensFault,  # Alias for consistency
    "pad_masking_error": PadMaskingErrorFault,
}


def create_decoder_masking_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int = 2,
    **kwargs
) -> AttentionFault:
    """
    Factory function to create decoder masking faults.

    Args:
        fault_type: Type of fault ("break_causal_mask", "over_mask_valid", "pad_masking_error")
        model: The decoder model
        layer_idx: Target layer index
        **kwargs: Additional fault-specific parameters

    Returns:
        Decoder masking fault instance

    Raises:
        ValueError: If fault_type is not recognized
    """
    if fault_type not in DECODER_MASKING_FAULTS:
        raise ValueError(
            f"Unknown decoder masking fault type: {fault_type}. "
            f"Available types: {list(DECODER_MASKING_FAULTS.keys())}"
        )

    fault_class = DECODER_MASKING_FAULTS[fault_type]
    return fault_class(model=model, layer_idx=layer_idx, **kwargs)
