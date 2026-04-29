"""Static and dynamic fault-injection mechanisms.

A :class:`FaultInjector` is a context manager. On enter it mutates the
target model in place; on exit it restores the original state. The two
mechanism classes are:

  StaticFault   modifies stored parameter tensors at rest. Examples:
                multiplying ``W1`` and ``W2`` of an FFN by a scalar
                factor, zeroing the query projection, scaling LayerNorm
                gamma. The injector backs up the original tensors before
                writing the mutated values, then writes the originals
                back on exit.
  DynamicFault  wraps the forward method of a target submodule (or the
                whole model) with a closure that intercepts the inputs,
                outputs, or kwargs of that call. Examples: zeroing the
                attention mask before softmax, breaking the causal mask
                in a decoder, returning stale K, V values from the cache.
                The injector saves the original ``forward`` reference
                and restores it on exit.

Subclasses implement ``_apply`` and ``_undo``. The base class provides
the context-manager glue plus a tiny utility (``_iter_target_modules``)
that resolves model sublayers by name and layer index.
"""
from __future__ import annotations

import abc
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn


class FaultInjector(AbstractContextManager, abc.ABC):
    """Abstract base for static and dynamic fault injectors.

    Subclasses must implement :meth:`_apply` and :meth:`_undo`. Both
    receive the bound model. The base ``__enter__`` and ``__exit__``
    enforce a single-shot lifecycle so that one injector instance is
    used for one paired (clean, faulty) run.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._applied = False

    # Context manager glue ────────────────────────────────────────────────
    def __enter__(self) -> "FaultInjector":
        if self._applied:
            raise RuntimeError(
                "This FaultInjector has already been entered. Construct a new "
                "instance per (clean, faulty) run pair to keep paired-seed "
                "comparisons clean.")
        self._apply(self.model)
        self._applied = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._applied:
            self._undo(self.model)
            self._applied = False
        return False  # do not suppress exceptions

    # Subclass interface ──────────────────────────────────────────────────
    @abc.abstractmethod
    def _apply(self, model: nn.Module) -> None:
        """Mutate ``model`` in place."""

    @abc.abstractmethod
    def _undo(self, model: nn.Module) -> None:
        """Restore ``model`` to its pre-injection state."""

    # Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _iter_target_modules(model: nn.Module,
                             layers: Iterable[int],
                             selector: Callable[[nn.Module, int], nn.Module | None]
                             ) -> list[tuple[int, nn.Module]]:
        """Resolve the per-layer submodules to mutate.

        Args:
            model:    full HF transformer.
            layers:   1-indexed layer indices to mutate.
            selector: function ``(model, layer_idx) -> submodule`` that
                      knows where the target submodule lives for this
                      architecture (e.g. encoder vs decoder, attention
                      block vs FFN). Returning ``None`` means "no target
                      at this layer".

        Returns:
            List of ``(layer_idx, submodule)`` pairs. Layers for which
            ``selector`` returns ``None`` are skipped silently (the
            structural verifier checks coverage separately).
        """
        out: list[tuple[int, nn.Module]] = []
        for idx in layers:
            sub = selector(model, idx)
            if sub is not None:
                out.append((idx, sub))
        return out


# ─────────────────────────────────────────────────────────────────────────
# Static injection: parameter backup and restore
# ─────────────────────────────────────────────────────────────────────────
class StaticFault(FaultInjector):
    """Modify parameter tensors at rest before forward execution.

    Subclasses override :meth:`mutate_parameters` to apply an in-place
    update to a list of parameter tensors. The base class records the
    original values and restores them on exit.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        # Backup map: id(param) -> (param_ref, original_clone).
        self._backups: dict[int, tuple[torch.nn.Parameter, torch.Tensor]] = {}

    def _backup(self, param: torch.nn.Parameter) -> None:
        if id(param) not in self._backups:
            self._backups[id(param)] = (param, param.detach().clone())

    def _apply(self, model: nn.Module) -> None:
        params = list(self.parameters_to_mutate(model))
        for p in params:
            self._backup(p)
        with torch.no_grad():
            self.mutate_parameters(params)

    def _undo(self, model: nn.Module) -> None:
        with torch.no_grad():
            for _, (param, original) in self._backups.items():
                param.data.copy_(original)
        self._backups.clear()

    # Subclass interface ──────────────────────────────────────────────────
    @abc.abstractmethod
    def parameters_to_mutate(self, model: nn.Module) -> Iterable[torch.nn.Parameter]:
        """Return the parameter tensors that this fault will mutate."""

    @abc.abstractmethod
    def mutate_parameters(self, params: list[torch.nn.Parameter]) -> None:
        """Mutate the listed parameters in place. Called under no_grad."""


# ─────────────────────────────────────────────────────────────────────────
# Dynamic injection: forward-method wrapping
# ─────────────────────────────────────────────────────────────────────────
class DynamicFault(FaultInjector):
    """Wrap the forward method of a target module with a closure.

    Subclasses override :meth:`target_modules` to return the list of
    modules whose ``forward`` should be replaced, and
    :meth:`make_faulty_forward` to build a closure that intercepts the
    call. The base class saves the original ``forward`` references and
    restores them on exit.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        # Backup: list of (module, original_forward).
        self._backups: list[tuple[nn.Module, Callable[..., Any]]] = []

    def _apply(self, model: nn.Module) -> None:
        for mod in self.target_modules(model):
            original = mod.forward
            self._backups.append((mod, original))
            mod.forward = self.make_faulty_forward(mod, original)

    def _undo(self, model: nn.Module) -> None:
        for mod, original in self._backups:
            mod.forward = original
        self._backups.clear()

    # Subclass interface ──────────────────────────────────────────────────
    @abc.abstractmethod
    def target_modules(self, model: nn.Module) -> list[nn.Module]:
        """Return the modules whose ``forward`` should be wrapped."""

    @abc.abstractmethod
    def make_faulty_forward(self, module: nn.Module,
                            original_forward: Callable[..., Any]
                            ) -> Callable[..., Any]:
        """Build a wrapped forward for ``module``.

        The returned callable is bound directly to ``module.forward``,
        so it should accept ``*args, **kwargs`` and call
        ``original_forward(*args, **kwargs)`` once it has applied the
        intended modification. ``self`` (the injector) may be captured
        via closure if the wrapper needs injector state.
        """
