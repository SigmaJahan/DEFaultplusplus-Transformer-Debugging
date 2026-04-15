"""
Shared utilities for attention mechanism fault injection.

Provides causal masking utilities and architecture detection for decoder models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def get_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate causal attention mask for decoder models.

    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type for mask

    Returns:
        Causal mask tensor of shape (seq_len, seq_len) with -inf for masked positions
    """
    # Create lower triangular matrix (1 for valid positions, 0 for masked)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))

    # Convert to additive mask (-inf for masked positions, 0 for valid)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)

    return mask


def is_causal_mask(mask: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """
    Check if a mask is causal (lower triangular structure).

    CRITICAL FIX: This function now properly handles different mask formats:
    - Additive masks with -inf for masked positions (explicit causal masks)
    - Binary masks with 0 for masked positions (padding-only masks in decoders)
    - Multi-dimensional masks (2D, 3D, 4D)

    For GPT-2 style decoders, the attention_mask is often just padding (1s and 0s),
    while the causal constraint is enforced internally. This is NOT a causal mask.

    Args:
        mask: Attention mask tensor
        tolerance: Tolerance for numerical comparison

    Returns:
        True if mask has explicit causal structure, False otherwise
    """
    if mask is None:
        return False

    # CRITICAL FIX: Handle different mask shapes consistently
    # Store original shape for later checks
    original_shape = mask.shape

    if mask.dim() == 2:
        # Shape: (seq_len, seq_len)
        check_mask = mask
    elif mask.dim() == 3:
        # Shape: (batch, seq_len, seq_len) - take first batch
        check_mask = mask[0]
    elif mask.dim() == 4:
        # Shape: (batch, heads, seq_len, seq_len) - take first batch, first head
        check_mask = mask[0, 0]
    else:
        return False

    seq_len = check_mask.shape[-1]

    # CRITICAL FIX: Detect padding-only masks (GPT-2 style)
    # These are binary masks (0s and 1s) or similar, NOT causal masks
    # The causal constraint is enforced internally in the model
    mask_values = check_mask.unique()
    if len(mask_values) <= 3 and check_mask.max() <= 1.0 and check_mask.min() >= 0.0:
        # This is a padding mask, not a causal mask
        return False

    # Check if upper triangle is masked (contains -inf or very negative values)
    upper_triangle = torch.triu(check_mask, diagonal=1)

    # Check if upper triangle values are all very negative (masked)
    # Account for both -inf and large negative values (e.g., -10000)
    # CRITICAL FIX: Use .item() to convert tensor bool to Python bool before 'or' operator
    is_upper_masked = (
        (upper_triangle == float('-inf')).all().item() or
        (upper_triangle < -1e4).all().item() or
        ((upper_triangle != 0) & (upper_triangle.abs() > 1e4)).all().item()
    )

    # Check if lower triangle + diagonal are valid (close to 0 or positive)
    lower_triangle_with_diag = torch.tril(check_mask)
    is_lower_valid = (lower_triangle_with_diag.abs() < 1e4).all().item()

    return is_upper_masked and is_lower_valid


def is_causal_architecture(model: nn.Module) -> bool:
    """
    Detect if model uses causal (decoder) attention.

    CRITICAL FIX: Expanded to include all common decoder architectures.
    Previously only checked: gpt2, gpt_neo, gpt_neox, gptj, llama, opt
    Now includes: All LLaMA variants, OPT, BLOOM, Falcon, Mistral, Qwen, Phi, etc.

    Args:
        model: PyTorch model

    Returns:
        True if model is a decoder architecture, False otherwise
    """
    if hasattr(model, 'config'):
        model_type = getattr(model.config, 'model_type', None)

        # Comprehensive list of decoder architectures
        decoder_types = [
            # Original supported types
            'gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'gpt-j',
            # LLaMA family
            'llama', 'llama2', 'llama3', 'codellama',
            # Meta models
            'opt', 'opt-iml',
            # BigScience
            'bloom', 'bloomz',
            # TII
            'falcon', 'refinedweb',
            # Mistral AI
            'mistral', 'mixtral',
            # Microsoft
            'phi', 'phi-2', 'phi-3',
            # Alibaba
            'qwen', 'qwen2',
            # Google
            'gemma', 'gemma2',
            # EleutherAI
            'pythia',
            # Stability AI
            'stablelm', 'stable-lm',
            # Other common decoders
            'mpt', 'redpajama', 'starcoder', 'santacoder',
            'gpt_bigcode', 'persimmon', 'yi'
        ]

        if model_type in decoder_types:
            return True

    # Fallback: check architecture structure
    # GPT-2 style: transformer.h
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return True

    # Pythia style: gpt_neox
    if hasattr(model, 'gpt_neox'):
        return True

    # LLaMA/OPT style: model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return True

    # Direct layer access
    if hasattr(model, 'layers'):
        return True

    return False


def preserve_causal_structure(
    original_mask: torch.Tensor,
    modified_mask: torch.Tensor
) -> torch.Tensor:
    """
    Ensure modified mask maintains causal structure where needed.

    If the original mask was causal, this ensures the modified mask
    also respects the causal constraint (no future token attention).

    Args:
        original_mask: Original attention mask
        modified_mask: Modified attention mask

        Returns:
        Modified mask with causal structure preserved
    """
    if original_mask is None:
        return modified_mask

    # Check if original was causal
    if not is_causal_mask(original_mask):
        return modified_mask

    # Ensure modified mask also respects causal constraint
    # by taking the minimum (more restrictive) of both masks
    return torch.minimum(modified_mask, original_mask)


def break_causal_mask(
    causal_mask: torch.Tensor,
    visibility_ratio: float = 0.5
) -> torch.Tensor:
    """
    Break causal mask by allowing attention to future positions.

    Args:
        causal_mask: Original causal mask
        visibility_ratio: Fraction of future positions to make visible (0.0 to 1.0)

    Returns:
        Modified mask with partial future visibility
    """
    if causal_mask is None:
        return None

    seq_len = causal_mask.shape[-1]
    device = causal_mask.device
    dtype = causal_mask.dtype

    # Create a mask for upper triangle
    upper_triangle_indices = torch.triu_indices(seq_len, seq_len, offset=1, device=device)

    # Randomly select which future positions to unmask
    num_upper = upper_triangle_indices.shape[1]
    num_to_unmask = int(num_upper * visibility_ratio)

    if num_to_unmask > 0:
        # Random indices to unmask
        perm = torch.randperm(num_upper, device=device)[:num_to_unmask]
        rows = upper_triangle_indices[0, perm]
        cols = upper_triangle_indices[1, perm]

        # Clone mask and unmask selected future positions
        broken_mask = causal_mask.clone()
        broken_mask[..., rows, cols] = 0.0  # 0.0 means not masked

    else:
        broken_mask = causal_mask

    return broken_mask


def over_mask_valid_tokens(
    causal_mask: torch.Tensor,
    mask_ratio: float = 0.2
) -> torch.Tensor:
    """
    Mask out some valid past tokens that should be visible.

    Args:
        causal_mask: Original causal mask
        mask_ratio: Fraction of valid past positions to mask out (0.0 to 1.0)

    Returns:
        Modified mask with some valid positions masked
    """
    if causal_mask is None:
        return None

    seq_len = causal_mask.shape[-1]
    device = causal_mask.device

    # Get strictly lower triangle indices (exclude diagonal to keep at least 1 valid token per row)
    lower_triangle_indices = torch.tril_indices(seq_len, seq_len, offset=-1, device=device)

    # Randomly select which valid positions to mask
    num_lower = lower_triangle_indices.shape[1]
    num_to_mask = int(num_lower * mask_ratio)

    if num_to_mask > 0:
        # Random indices to mask
        perm = torch.randperm(num_lower, device=device)[:num_to_mask]
        rows = lower_triangle_indices[0, perm]
        cols = lower_triangle_indices[1, perm]

        # Clone mask and mask selected valid positions
        over_masked = causal_mask.clone()
        over_masked[..., rows, cols] = float('-inf')  # -inf means masked

    else:
        over_masked = causal_mask

    return over_masked


def get_attention_module_from_layer(layer: nn.Module) -> Optional[nn.Module]:
    """
    Extract attention module from a transformer layer.

    Handles both encoder and decoder architectures.

    Args:
        layer: Transformer layer module

    Returns:
        Attention module or None if not found
    """
    # GPT-2 / DistilGPT2: layer.attn
    if hasattr(layer, 'attn'):
        return layer.attn

    # Pythia / GPT-NeoX: layer.attention
    if hasattr(layer, 'attention'):
        return layer.attention

    # BERT / RoBERTa: layer.attention
    if hasattr(layer, 'self_attn'):
        return layer.self_attn

    # DistilBERT: layer.attention
    if hasattr(layer, 'attention'):
        return layer.attention

    return None


def get_qkv_projections(attention_module: nn.Module) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
    """
    Extract Q, K, V projection layers from attention module.

    Args:
        attention_module: Attention module

    Returns:
        Tuple of (q_proj, k_proj, v_proj) or (None, None, None) if not found
    """
    # GPT-2 style: c_attn (combined QKV) or separate q_attn, k_attn, v_attn
    if hasattr(attention_module, 'c_attn'):
        # Combined QKV projection
        return attention_module.c_attn, attention_module.c_attn, attention_module.c_attn

    if hasattr(attention_module, 'q_attn') and hasattr(attention_module, 'k_attn') and hasattr(attention_module, 'v_attn'):
        return attention_module.q_attn, attention_module.k_attn, attention_module.v_attn

    # Pythia / GPT-NeoX style
    if hasattr(attention_module, 'query_key_value'):
        # Combined QKV
        qkv = attention_module.query_key_value
        return qkv, qkv, qkv

    # Separate projections
    q_proj = getattr(attention_module, 'q_proj', None) or getattr(attention_module, 'q_lin', None)
    k_proj = getattr(attention_module, 'k_proj', None) or getattr(attention_module, 'k_lin', None)
    v_proj = getattr(attention_module, 'v_proj', None) or getattr(attention_module, 'v_lin', None)

    return q_proj, k_proj, v_proj
