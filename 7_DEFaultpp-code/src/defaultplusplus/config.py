"""Configuration dataclasses for DEFault++."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""

    # Gradient thresholds
    grad_vanish_threshold: float = 1e-4
    grad_explode_threshold: float = 100.0
    grad_activity_threshold: float = 1e-6

    # Attention
    attention_leak_threshold: float = 1e-6
    position_cutoff: int = 64
    ece_num_bins: int = 15

    # Structural probes
    ffn_probe_tokens: int = 256
    ffn_var_activity_threshold: float = 1e-6

    # Collection frequency
    activation_interval: int = 10
    gradient_window: int = 20

    # Representation drift
    representation_epochs: List[int] = field(default_factory=lambda: [1, 4, 7, 10])
    representation_tokens: int = 256

    # Special token IDs (auto-detected from tokenizer if None)
    pad_token_id: Optional[int] = None
    cls_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
