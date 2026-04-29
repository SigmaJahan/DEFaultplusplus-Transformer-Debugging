"""Forward-hook plumbing for sublayer-boundary tensor capture.

The :class:`SublayerCapture` registers PyTorch forward hooks on every
discovered layer's attention, FFN, and LayerNorm submodules — plus the
Q/K/V projection ``Linear``\\s inside each attention block — so that
``StructuralMetrics`` and ``AttentionMetrics`` can read the *exact*
sublayer-boundary tensors instead of differencing adjacent hidden
states.

One capture object is constructed per :class:`FeatureExtractor` and
lives for the duration of the run. Each forward pass overwrites the
previous step's tensors; metric modules consume them in
``collect_step`` and the capture clears between steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .inspector import ModelInspector


# Keys in ``captures`` are (layer_idx, tap_name).
CaptureKey = Tuple[int, str]
CaptureDict = Dict[CaptureKey, torch.Tensor]


class SublayerCapture:
    """Register forward hooks for sublayer-boundary tensor capture.

    Hooks tap five sites per layer:

      ``attn_in`` / ``attn_out``      attention sublayer input / output
      ``ffn_in`` / ``ffn_out``        FFN sublayer input / output
      ``ln_pre_attn_in/out`` /
      ``ln_pre_ffn_in/out``           per-LayerNorm input / output
      ``q`` / ``k`` / ``v``           post-projection query / key / value
                                       tensors (separate-style attention)

    Tensors are stored as ``.detach()`` references; consumers must
    ``.clone()`` if they intend to mutate. Captures live for the most
    recent forward pass only — call :meth:`clear` between steps.
    """

    def __init__(self, inspector: ModelInspector) -> None:
        self.inspector = inspector
        self._handles: List[Any] = []
        self.captures: CaptureDict = {}
        self._installed = False

    # ── Lifecycle ────────────────────────────────────────────────────────
    def install(self) -> None:
        """Register all forward hooks. Idempotent."""
        if self._installed:
            return

        for layer_idx, layer in enumerate(self.inspector.layers):
            self._install_attention(layer_idx, layer)
            self._install_ffn(layer_idx, layer)
            self._install_layernorms(layer_idx, layer)

        self._installed = True

    def remove(self) -> None:
        """Deregister all hooks and drop captured tensors."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # pragma: no cover - defensive
                pass
        self._handles.clear()
        self.captures.clear()
        self._installed = False

    def clear(self) -> None:
        """Drop captured tensors but keep hooks installed."""
        self.captures.clear()

    def __enter__(self) -> "SublayerCapture":
        self.install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.remove()
        return False

    # ── Convenience accessors ────────────────────────────────────────────
    def get(self, layer_idx: int, tap: str) -> Optional[torch.Tensor]:
        return self.captures.get((layer_idx, tap))

    def has(self, layer_idx: int, tap: str) -> bool:
        return (layer_idx, tap) in self.captures

    @property
    def installed(self) -> bool:
        return self._installed

    # ── Hook installation ────────────────────────────────────────────────
    def _install_attention(self, layer_idx: int, layer: nn.Module) -> None:
        attn_attr = self.inspector._attn_pattern.module_attr if self.inspector._attn_pattern else None
        attn_module = getattr(layer, attn_attr, None) if attn_attr else None
        if attn_module is None:
            return

        self._handles.append(
            attn_module.register_forward_pre_hook(
                self._make_pre_input_hook(layer_idx, "attn_in")
            )
        )
        self._handles.append(
            attn_module.register_forward_hook(
                self._make_post_output_hook(layer_idx, "attn_out")
            )
        )

        # Q/K/V projection taps.
        qkv = self.inspector.get_qkv_modules(layer_idx)
        if not qkv:
            return
        names = self.inspector._attn_pattern.qkv_names
        if self.inspector._attn_pattern.qkv_style == "separate" and len(qkv) >= 3:
            for tap, mod in zip(("q", "k", "v"), qkv[:3]):
                self._handles.append(
                    mod.register_forward_hook(
                        self._make_post_output_hook(layer_idx, tap)
                    )
                )
        elif self.inspector._attn_pattern.qkv_style == "fused" and len(qkv) == 1:
            # Fused QKV: capture the whole tensor under tap "qkv_fused".
            # Splitting requires per-architecture knowledge; downstream
            # consumers fall back to reconstructed metrics in that case.
            self._handles.append(
                qkv[0].register_forward_hook(
                    self._make_post_output_hook(layer_idx, "qkv_fused")
                )
            )

    def _install_ffn(self, layer_idx: int, layer: nn.Module) -> None:
        ffn_attr = self.inspector._ffn_pattern.module_attr if self.inspector._ffn_pattern else None
        ffn_module = getattr(layer, ffn_attr, None) if ffn_attr else None
        if ffn_module is None:
            return

        self._handles.append(
            ffn_module.register_forward_pre_hook(
                self._make_pre_input_hook(layer_idx, "ffn_in")
            )
        )
        self._handles.append(
            ffn_module.register_forward_hook(
                self._make_post_output_hook(layer_idx, "ffn_out")
            )
        )

    def _install_layernorms(self, layer_idx: int, layer: nn.Module) -> None:
        ln_names = self.inspector._ln_names or []
        if not ln_names:
            return

        # Map LN attr name to a stable tap label. When the layer reports
        # multiple LayerNorms we order them and call them ``ln0``, ``ln1``
        # … so downstream consumers can iterate without depending on
        # arch-specific names.
        for ord_idx, ln_attr in enumerate(ln_names):
            ln_module = _resolve_dotted(layer, ln_attr)
            if ln_module is None:
                continue
            tap_in = f"ln{ord_idx}_in"
            tap_out = f"ln{ord_idx}_out"
            self._handles.append(
                ln_module.register_forward_pre_hook(
                    self._make_pre_input_hook(layer_idx, tap_in)
                )
            )
            self._handles.append(
                ln_module.register_forward_hook(
                    self._make_post_output_hook(layer_idx, tap_out)
                )
            )

    # ── Hook factories ───────────────────────────────────────────────────
    def _make_pre_input_hook(self, layer_idx: int, tap: str):
        def _hook(_module: nn.Module, args: tuple) -> None:
            if not args:
                return
            tensor = args[0]
            if torch.is_tensor(tensor):
                self.captures[(layer_idx, tap)] = tensor.detach()
        return _hook

    def _make_post_output_hook(self, layer_idx: int, tap: str):
        def _hook(_module: nn.Module, _inputs: tuple, output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                self.captures[(layer_idx, tap)] = tensor.detach()
        return _hook


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """Return the first ``torch.Tensor`` in nested tuples/lists/objects."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _first_tensor(item)
            if t is not None:
                return t
        return None
    if hasattr(obj, "last_hidden_state") and torch.is_tensor(obj.last_hidden_state):
        return obj.last_hidden_state
    if hasattr(obj, "logits") and torch.is_tensor(obj.logits):
        return obj.logits
    return None


def _resolve_dotted(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    """Resolve ``a.b.c`` against ``root``; return ``None`` if any step misses."""
    obj: Any = root
    for part in dotted.split("."):
        if not part:
            continue
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, nn.Module) else None


__all__ = ["SublayerCapture"]
