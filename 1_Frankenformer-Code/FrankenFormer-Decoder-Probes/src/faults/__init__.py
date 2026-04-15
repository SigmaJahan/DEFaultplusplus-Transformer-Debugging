"""Fault injection modules for encoder and decoder models."""

# Base classes
from src.faults.base_fault import BaseFault, AttentionFault

# Decoder-specific faults
from src.faults.decoder_masking_faults import (
    BreakCausalMaskFault,
    OverMaskValidTokensFault,
    PadMaskingErrorFault,
    create_decoder_masking_fault,
    DECODER_MASKING_FAULTS,
)

from src.faults.kv_cache_faults import (
    StaleCacheFault,
    OffByOneIndexFault,
    TruncatedCacheFault,
    CrossRequestCacheLeakFault,
    create_kv_cache_fault,
    KV_CACHE_FAULTS,
)

# Shared utilities
from src.faults.attention_utils import (
    get_causal_mask,
    is_causal_mask,
    is_causal_architecture,
    break_causal_mask,
    over_mask_valid_tokens,
)

__all__ = [
    # Base classes
    "BaseFault",
    "AttentionFault",
    # Decoder masking faults
    "BreakCausalMaskFault",
    "OverMaskValidTokensFault",
    "PadMaskingErrorFault",
    "create_decoder_masking_fault",
    "DECODER_MASKING_FAULTS",
    # KV-cache faults
    "StaleCacheFault",
    "OffByOneIndexFault",
    "TruncatedCacheFault",
    "CrossRequestCacheLeakFault",
    "create_kv_cache_fault",
    "KV_CACHE_FAULTS",
    # Utilities
    "get_causal_mask",
    "is_causal_mask",
    "is_causal_architecture",
    "break_causal_mask",
    "over_mask_valid_tokens",
]
