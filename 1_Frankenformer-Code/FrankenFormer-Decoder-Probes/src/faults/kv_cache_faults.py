"""
KV-Cache Management Faults (Decoder-Only)

Implements four types of KV-cache faults specific to decoder models:
- Stale Cache: Don't update cache for some layers/steps
- Off-By-One Index: Cache indexing shifted by ±1
- Truncated Cache: Drop recent cache blocks for long sequences
- Cross-Request Cache Leak: Fail to clear cache between prompts

These faults target the key-value cache mechanism used in autoregressive generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple, List
from src.faults.base_fault import BaseFault


class StaleCacheFault(BaseFault):
    """
    Stale Cache Fault

    Prevents cache updates for specified layers or after a certain position,
    causing the model to reuse stale cached key-value pairs.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        freeze_after: int = 100,
        affected_layers: Optional[List[int]] = None,
    ):
        """
        Initialize stale cache fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Primary target layer index (for single-layer faults)
            freeze_after: Position after which to freeze cache updates
            affected_layers: List of layer indices to freeze, or None for single layer
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="stale_cache",
            description=f"Freeze cache updates after position {freeze_after}"
        )
        self.freeze_after = freeze_after
        self.affected_layers = affected_layers or [layer_idx]
        self.cached_kv: dict = {}
        self.step_count = 0

    def inject(self) -> None:
        """Inject the stale cache fault."""
        # Hook into model's forward to intercept cache
        self._backup_forward = self.model.forward

        freeze_after = self.freeze_after
        affected_layers = self.affected_layers
        cached_kv = self.cached_kv
        step_count_holder = {'count': 0}

        def faulty_forward(
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs
        ):
            """Forward with stale cache."""
            step_count_holder['count'] += 1

            # If we've passed the freeze point and have cached KV, use stale cache
            if step_count_holder['count'] > freeze_after and cached_kv:
                # Use frozen cache instead of current cache
                frozen_past = []
                for layer_idx in range(len(past_key_values) if past_key_values else 0):
                    if layer_idx in affected_layers and layer_idx in cached_kv:
                        # Use stale cache for this layer
                        frozen_past.append(cached_kv[layer_idx])
                    else:
                        # Use current cache
                        frozen_past.append(past_key_values[layer_idx] if past_key_values else None)

                past_key_values = tuple(frozen_past) if frozen_past else past_key_values

            # Call original forward
            outputs = self._backup_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

            # Cache KV at freeze point
            if step_count_holder['count'] == freeze_after and hasattr(outputs, 'past_key_values'):
                if outputs.past_key_values:
                    for layer_idx in affected_layers:
                        if layer_idx < len(outputs.past_key_values):
                            cached_kv[layer_idx] = outputs.past_key_values[layer_idx]

            return outputs

        self.model.forward = faulty_forward
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} - will freeze cache after position {freeze_after}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            self.model.forward = self._backup_forward

        self._backup_forward = None
        self.cached_kv.clear()
        self.step_count = 0
        self.is_injected = False

        self._log_info(f"Restored from {self.fault_name}")


class OffByOneIndexFault(BaseFault):
    """
    Off-By-One Index Fault

    Shifts cache read/write indices by ±1, causing misalignment between
    current position and cached key-value pairs.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        offset: int = 1,
        mode: str = "read",
        affected_layers: Optional[List[int]] = None,
    ):
        """
        Initialize off-by-one index fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Primary target layer index
            offset: Index offset (+1 or -1)
            mode: "read", "write", or "both"
            affected_layers: List of layer indices to affect
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="off_by_one_index",
            description=f"Cache index offset by {offset} ({mode} mode)"
        )
        self.offset = offset
        self.mode = mode
        self.affected_layers = affected_layers or [layer_idx]

    def inject(self) -> None:
        """Inject the off-by-one index fault."""
        # Hook into model's forward
        self._backup_forward = self.model.forward

        offset = self.offset
        mode = self.mode
        affected_layers = self.affected_layers

        def faulty_forward(
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs
        ):
            """Forward with off-by-one cache indexing."""
            # Modify past_key_values if in read mode
            if past_key_values and mode in ["read", "both"]:
                modified_past = []
                for layer_idx, layer_cache in enumerate(past_key_values):
                    if layer_idx in affected_layers and layer_cache is not None:
                        # Shift cache indices
                        # layer_cache is typically (key, value) tuple
                        # Each is shape [batch, num_heads, seq_len, head_dim]
                        key, value = layer_cache

                        # Roll along sequence dimension (dim=2)
                        shifted_key = torch.roll(key, shifts=offset, dims=2)
                        shifted_value = torch.roll(value, shifts=offset, dims=2)

                        modified_past.append((shifted_key, shifted_value))
                    else:
                        modified_past.append(layer_cache)

                past_key_values = tuple(modified_past)

            # Call original forward
            outputs = self._backup_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

            # Modify output cache if in write mode
            if mode in ["write", "both"] and hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                modified_output_past = []
                for layer_idx, layer_cache in enumerate(outputs.past_key_values):
                    if layer_idx in affected_layers and layer_cache is not None:
                        key, value = layer_cache
                        shifted_key = torch.roll(key, shifts=offset, dims=2)
                        shifted_value = torch.roll(value, shifts=offset, dims=2)
                        modified_output_past.append((shifted_key, shifted_value))
                    else:
                        modified_output_past.append(layer_cache)

                # Create new outputs with modified cache
                outputs.past_key_values = tuple(modified_output_past)

            return outputs

        self.model.forward = faulty_forward
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} with offset {offset} in {mode} mode")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            self.model.forward = self._backup_forward

        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored from {self.fault_name}")


class TruncatedCacheFault(BaseFault):
    """
    Truncated Cache Fault

    Drops the most recent cache entries for sequences exceeding a threshold,
    simulating incorrect cache management for long sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        truncate_last: int = 5,
        trigger_length: int = 100,
    ):
        """
        Initialize truncated cache fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Primary target layer index
            truncate_last: Number of recent positions to drop
            trigger_length: Sequence length that triggers truncation
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="truncated_cache",
            description=f"Drop last {truncate_last} cache positions when seq > {trigger_length}"
        )
        self.truncate_last = truncate_last
        self.trigger_length = trigger_length

    def inject(self) -> None:
        """Inject the truncated cache fault."""
        self._backup_forward = self.model.forward

        truncate_last = self.truncate_last
        trigger_length = self.trigger_length

        def faulty_forward(
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs
        ):
            """Forward with truncated cache."""
            # Call original forward first
            outputs = self._backup_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

            # Truncate output cache if sequence is long enough
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                # Check cache length
                first_layer_cache = outputs.past_key_values[0]
                if first_layer_cache is not None:
                    cache_len = first_layer_cache[0].shape[2]  # seq_len dimension

                    if cache_len > trigger_length:
                        # Truncate last N positions
                        truncated_past = []
                        for layer_cache in outputs.past_key_values:
                            if layer_cache is not None:
                                key, value = layer_cache
                                # Keep all but last truncate_last positions
                                truncated_key = key[:, :, :-truncate_last, :]
                                truncated_value = value[:, :, :-truncate_last, :]
                                truncated_past.append((truncated_key, truncated_value))
                            else:
                                truncated_past.append(None)

                        outputs.past_key_values = tuple(truncated_past)

            return outputs

        self.model.forward = faulty_forward
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name}")

    def restore(self) -> None:
        """Restore original forward method."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            self.model.forward = self._backup_forward

        self._backup_forward = None
        self.is_injected = False

        self._log_info(f"Restored from {self.fault_name}")


class CrossRequestCacheLeakFault(BaseFault):
    """
    Cross-Request Cache Leak Fault

    Fails to properly clear cache between independent prompts,
    causing context leakage across different requests.
    """

    # Memory safety: Limit maximum leaked cache to prevent OOM on Compute Canada GPUs
    MAX_LEAKED_CACHE_TOKENS = 256  # Cap at 256 tokens to ensure <12GB VRAM usage

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 2,
        leak_ratio: float = 0.7,
        num_layers: Optional[int] = None,
    ):
        """
        Initialize cross-request cache leak fault.

        Args:
            model: The decoder model to inject fault into
            layer_idx: Primary target layer index
            leak_ratio: Fraction of cache to retain between requests (0.0 to 1.0)
            num_layers: Number of layers to leak cache from, or None for all
        """
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="cross_request_leak",
            description=f"Retain {leak_ratio*100:.0f}% of cache between requests"
        )
        self.leak_ratio = leak_ratio
        self.num_layers = num_layers
        self.leaked_cache: Optional[Tuple] = None
        self.leaked_cache_holder: Optional[dict] = None  # Store reference for cleanup

    def inject(self) -> None:
        """Inject the cross-request cache leak fault."""
        # Hook into model's forward
        self._backup_forward = self.model.forward

        leak_ratio = self.leak_ratio
        num_layers = self.num_layers
        leaked_cache_holder = {'cache': None}

        def faulty_forward(
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=None,
            **kwargs
        ):
            """Forward with leaked cache from previous request."""
            # MEMORY SAFETY FIX #1: Check GPU memory and auto-clear if approaching OOM
            if torch.cuda.is_available() and leaked_cache_holder['cache'] is not None:
                mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
                # Conservative threshold: Clear cache if using >10GB (safe for 16GB GPUs)
                if mem_reserved_gb > 10.0:
                    leaked_cache_holder['cache'] = None
                    self._log_info(f"Auto-cleared leaked cache (mem usage: {mem_reserved_gb:.1f}GB)")

            # If this is a new request (no past_key_values) but we have leaked cache,
            # inject the leaked cache
            if past_key_values is None and leaked_cache_holder['cache'] is not None:
                # MEMORY SAFETY FIX #2: Validate batch size matches
                first_layer_cache = leaked_cache_holder['cache'][0]
                if first_layer_cache is not None:
                    leaked_batch_size = first_layer_cache[0].shape[0]
                    current_batch_size = input_ids.shape[0] if input_ids is not None else 0

                    if leaked_batch_size != current_batch_size:
                        # Batch size mismatch - clear incompatible cache to prevent CUDA errors
                        leaked_cache_holder['cache'] = None
                        self._log_info(f"Cleared leaked cache (batch mismatch: {leaked_batch_size} vs {current_batch_size})")
                    else:
                        # Use (part of) the leaked cache
                        past_key_values = leaked_cache_holder['cache']

                        # CRITICAL FIX (Bug #15): Extend attention_mask to account for leaked cache positions
                        # Without this, tensor dimension mismatch occurs during attention computation
                        if attention_mask is not None:
                            cache_len = first_layer_cache[0].shape[2]  # seq_len dimension

                            # Extend attention_mask by prepending 1s for the leaked cache positions
                            batch_size = attention_mask.shape[0]
                            # Create mask for leaked cache (all 1s, meaning attend to leaked positions)
                            cache_mask = torch.ones(
                                batch_size, cache_len,
                                dtype=attention_mask.dtype,
                                device=attention_mask.device
                            )
                            # Concatenate: [cache_mask, original_attention_mask]
                            attention_mask = torch.cat([cache_mask, attention_mask], dim=1)

            # Call original forward
            outputs = self._backup_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

            # Store cache for potential leak to next request
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                # Decide which layers to leak
                layers_to_leak = num_layers if num_layers is not None else len(outputs.past_key_values)

                leaked_cache = []
                for layer_idx, layer_cache in enumerate(outputs.past_key_values):
                    if layer_idx < layers_to_leak and layer_cache is not None:
                        key, value = layer_cache

                        # MEMORY SAFETY FIX #3: Cap leaked cache size to prevent unbounded growth
                        cache_len = key.shape[2]
                        retain_len = max(1, int(cache_len * leak_ratio))
                        # Apply maximum limit to prevent OOM
                        retain_len = min(retain_len, CrossRequestCacheLeakFault.MAX_LEAKED_CACHE_TOKENS)

                        # Keep last retain_len positions
                        leaked_key = key[:, :, -retain_len:, :].clone().detach()
                        leaked_value = value[:, :, -retain_len:, :].clone().detach()

                        leaked_cache.append((leaked_key, leaked_value))
                    else:
                        leaked_cache.append(None)

                leaked_cache_holder['cache'] = tuple(leaked_cache)

            return outputs

        self.model.forward = faulty_forward
        # Store reference to cache holder for cleanup in restore()
        self.leaked_cache_holder = leaked_cache_holder
        self.is_injected = True

        self._log_info(f"Injected {self.fault_name} with {leak_ratio*100:.0f}% cache retention (max {self.MAX_LEAKED_CACHE_TOKENS} tokens)")

    def restore(self) -> None:
        """Restore original forward method and clear leaked cache."""
        if not self.is_injected:
            return

        if self._backup_forward is not None:
            self.model.forward = self._backup_forward

        # MEMORY SAFETY FIX #4: Explicitly clear leaked cache to free GPU memory
        if self.leaked_cache_holder is not None:
            self.leaked_cache_holder['cache'] = None
            self.leaked_cache_holder = None

        # Force GPU cache clear for immediate memory reclamation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._backup_forward = None
        self.leaked_cache = None
        self.is_injected = False

        self._log_info(f"Restored from {self.fault_name} and cleared leaked cache")


# Registry of KV-cache faults
KV_CACHE_FAULTS = {
    "stale_cache": StaleCacheFault,
    "off_by_one_index": OffByOneIndexFault,
    "truncated_cache": TruncatedCacheFault,
    "cross_request_leak": CrossRequestCacheLeakFault,
    "cross_request_cache_leak": CrossRequestCacheLeakFault,  # Alias for consistency
}


def create_kv_cache_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int = 2,
    **kwargs
) -> BaseFault:
    """
    Factory function to create KV-cache faults.

    Args:
        fault_type: Type of fault ("stale_cache", "off_by_one_index", "truncated_cache", "cross_request_leak")
        model: The decoder model
        layer_idx: Target layer index
        **kwargs: Additional fault-specific parameters

    Returns:
        KV-cache fault instance

    Raises:
        ValueError: If fault_type is not recognized
    """
    if fault_type not in KV_CACHE_FAULTS:
        raise ValueError(
            f"Unknown KV-cache fault type: {fault_type}. "
            f"Available types: {list(KV_CACHE_FAULTS.keys())}"
        )

    fault_class = KV_CACHE_FAULTS[fault_type]
    return fault_class(model=model, layer_idx=layer_idx, **kwargs)
