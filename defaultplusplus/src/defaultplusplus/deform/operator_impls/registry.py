"""Factory and implementations for the 52 DEForm operators.

The implementations are intentionally architecture-tolerant: they target
standard HuggingFace naming conventions first and fail explicitly when a
static operator has no compatible parameter on the supplied model.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..injection import DynamicFault, FaultInjector, StaticFault
from ..operators import OPERATORS


_MASK_OPS = {"MZM", "MIM", "MRM", "MCB"}
_SCORE_OPS = {"SDS", "SPD", "SUC"}
_POSITIONAL_OPS = {"POE", "PSI", "PTL"}
_KERNEL_OPS = {"KSB", "KMD", "KFT", "KRP"}
_VARIANT_OPS = {"VSH", "VEC"}
_CACHE_OPS = {"CST", "CDU", "COB", "CTR", "CLK"}
_RESIDUAL_OPS = {"RRS", "RSR", "RIN"}
_OUTPUT_DYNAMIC_OPS = {"OSL", "OOD"}
_ATTRIBUTE_OPS = {"QFG", "QHD", "FCA", "FRG", "NCE", "NWD", "RGC", "RDR", "KMC"}


@dataclass(frozen=True)
class StaticSpec:
    patterns: tuple[str, ...]
    mutator: str


_STATIC_SPECS: dict[str, StaticSpec] = {
    "QZQ": StaticSpec(("query", "q_proj", "q_lin", ".q.", "_q.", "self.q"), "zero"),
    "QZK": StaticSpec(("key", "k_proj", "k_lin", ".k.", "_k.", "self.k"), "zero"),
    "QZV": StaticSpec(("value", "v_proj", "v_lin", ".v.", "_v.", "self.v"), "zero"),
    "QSW": StaticSpec(("query", "q_proj", "q_lin", "key", "k_proj", "k_lin"), "swap_qk"),
    "QTH": StaticSpec(("query", "q_proj", "q_lin", "key", "k_proj", "k_lin",
                       "value", "v_proj", "v_lin"), "tie_rows"),
    "ETZ": StaticSpec(("embeddings.word_embeddings", "word_embedding", "tok_embedding",
                       "token_embedding", "wte", "embed_tokens"), "zero_rows"),
    "ESW": StaticSpec(("embeddings.word_embeddings", "word_embedding", "tok_embedding",
                       "token_embedding", "wte", "embed_tokens"), "swap_rows"),
    "ESS": StaticSpec(("token_type_embeddings", "segment", "type_embedding"), "scale"),
    "EZD": StaticSpec(("embeddings.word_embeddings", "word_embedding", "tok_embedding",
                       "token_embedding", "wte", "embed_tokens"), "zero_cols"),
    "FSW": StaticSpec(("intermediate.dense", "output.dense", "mlp", "ffn", "feed_forward",
                       "fc1", "fc2", "c_fc", "c_proj", "w1", "w2"), "scale"),
    "FDN": StaticSpec(("intermediate.dense", "mlp", "ffn", "feed_forward", "fc1",
                       "c_fc", "w1"), "zero_rows"),
    "FWI": StaticSpec(("intermediate.dense", "output.dense", "mlp", "ffn", "feed_forward",
                       "fc1", "fc2", "c_fc", "c_proj", "w1", "w2"), "init"),
    "NSG": StaticSpec(("layernorm", "layer_norm", "ln_", ".ln", "norm"), "scale"),
    "NZG": StaticSpec(("layernorm", "layer_norm", "ln_", ".ln", "norm"), "zero"),
    "NSB": StaticSpec(("layernorm", "layer_norm", "ln_", ".ln", "norm"), "shift"),
    "OZR": StaticSpec(("lm_head", "classifier", "score", "output_projection",
                       "predictions.decoder"), "zero_rows"),
    "ORI": StaticSpec(("lm_head", "classifier", "score", "output_projection",
                       "predictions.decoder"), "init"),
}


_DYNAMIC_PATTERNS: dict[str, tuple[str, ...]] = {
    **{op: ("attention", "attn", "self_attn") for op in _MASK_OPS | _SCORE_OPS | _VARIANT_OPS},
    **{op: ("position", "embed_positions", "wpe") for op in _POSITIONAL_OPS},
    **{op: ("attention", "attn", "self_attn") for op in _KERNEL_OPS},
    **{op: ("layer", "block", "encoder", "decoder") for op in _RESIDUAL_OPS},
    **{op: ("lm_head", "classifier", "score", "output_projection") for op in _OUTPUT_DYNAMIC_OPS},
    # Cache operators wrap the model forward (not the per-layer attention)
    # so they see the whole DynamicCache object once per forward and can
    # mutate the per-layer slices selectively.
    **{op: () for op in _CACHE_OPS},
}


def get_injector(operator_id: str,
                 *,
                 layers: Sequence[int] = (),
                 param_value: Any | None = None,
                 severity: str | None = None,
                 ) -> type[FaultInjector] | Callable[[nn.Module], FaultInjector]:
    """Return an injector constructor for a DEForm operator ID."""
    op_id = operator_id.upper()
    if op_id not in OPERATORS:
        raise KeyError(f"Unknown DEForm operator id: {operator_id!r}")

    value = _default_param_value(op_id, param_value)
    if op_id in _STATIC_SPECS:
        return _make_static_class(op_id, _STATIC_SPECS[op_id], tuple(layers), value)
    if op_id in _ATTRIBUTE_OPS:
        return _make_attribute_class(op_id, tuple(layers), value, severity)
    if op_id in _DYNAMIC_PATTERNS:
        return _make_dynamic_class(op_id, tuple(layers), value)
    raise KeyError(f"No injector implementation registered for {op_id}")


def get_expected_parameter_names(model: nn.Module, injector: FaultInjector) -> list[str]:
    """Return static parameter names expected to change for verifier use."""
    if hasattr(injector, "expected_parameter_names"):
        return list(injector.expected_parameter_names(model))  # type: ignore[attr-defined]
    return []


def get_expected_modules(model: nn.Module, injector: FaultInjector) -> list[nn.Module]:
    """Return dynamic target modules expected to be wrapped for verifier use."""
    if isinstance(injector, DynamicFault):
        return injector.target_modules(model)
    return []


def _default_param_value(op_id: str, explicit: Any | None) -> Any | None:
    if explicit is not None:
        return explicit
    grid = OPERATORS[op_id].param_grid
    return grid[0] if grid else None


def _make_static_class(op_id: str,
                       spec: StaticSpec,
                       layers: tuple[int, ...],
                       value: Any | None) -> type[StaticFault]:
    class _OperatorStaticFault(_NamedParameterFault):
        operator_id = op_id
        patterns = spec.patterns
        mutator = spec.mutator
        target_layers = layers
        param_value = value

    _OperatorStaticFault.__name__ = f"{op_id}Injector"
    return _OperatorStaticFault


def _make_dynamic_class(op_id: str,
                        layers: tuple[int, ...],
                        value: Any | None) -> type[DynamicFault]:
    class _OperatorDynamicFault(_ForwardFault):
        operator_id = op_id
        patterns = _DYNAMIC_PATTERNS[op_id]
        target_layers = layers
        param_value = value

    _OperatorDynamicFault.__name__ = f"{op_id}Injector"
    return _OperatorDynamicFault


def _make_attribute_class(op_id: str,
                          layers: tuple[int, ...],
                          value: Any | None,
                          severity: str | None) -> type[FaultInjector]:
    sev = severity
    base: type[FaultInjector]
    if op_id == "QFG":
        base = _RequiresGradFault
    elif op_id == "FCA":
        base = _ActivationFault
    elif op_id == "NCE":
        base = _LayerNormEpsilonFault
    else:
        base = _ModelAttributeFault

    class _OperatorAttributeFault(base):  # type: ignore[misc, valid-type]
        operator_id = op_id
        target_layers = layers
        param_value = value
        severity = sev

    _OperatorAttributeFault.__name__ = f"{op_id}Injector"
    return _OperatorAttributeFault


class _NamedParameterFault(StaticFault):
    operator_id: str
    patterns: tuple[str, ...]
    mutator: str
    target_layers: tuple[int, ...]
    param_value: Any | None

    def expected_parameter_names(self, model: nn.Module) -> list[str]:
        return [name for name, _ in self._named_targets(model)]

    def parameters_to_mutate(self, model: nn.Module) -> Iterable[torch.nn.Parameter]:
        targets = self._named_targets(model)
        if not targets:
            raise ValueError(f"{self.operator_id} found no compatible parameters")
        self._last_target_names = [name for name, _ in targets]
        return [p for _, p in targets]

    def mutate_parameters(self, params: list[torch.nn.Parameter]) -> None:
        if self.mutator == "zero":
            for p in params:
                p.zero_()
        elif self.mutator == "scale":
            factor = float(self.param_value if self.param_value is not None else 0.5)
            for p in params:
                p.mul_(factor)
        elif self.mutator == "shift":
            shift = float(self.param_value if self.param_value is not None else 0.5)
            for p in params:
                p.add_(shift)
        elif self.mutator == "zero_rows":
            frac = _fraction(self.param_value)
            for p in params:
                _zero_rows(p, frac)
        elif self.mutator == "zero_cols":
            frac = _fraction(self.param_value)
            for p in params:
                _zero_cols(p, frac)
        elif self.mutator == "swap_rows":
            frac = _fraction(self.param_value)
            for p in params:
                _swap_rows(p, frac)
        elif self.mutator == "swap_qk":
            names = getattr(self, "_last_target_names", None) or []
            if len(names) != len(params):
                raise ValueError(
                    f"{self.operator_id}: target name/param count mismatch "
                    f"({len(names)} vs {len(params)})"
                )
            swapped = _swap_qk_within_blocks(list(zip(names, params)))
            if swapped == 0:
                raise ValueError(
                    f"{self.operator_id} found Q/K targets but no swappable "
                    f"(Q,K) pair of equal shape within the same attention block"
                )
        elif self.mutator == "tie_rows":
            for p in params:
                _tie_rows(p)
        elif self.mutator == "init":
            for p in params:
                _initialize(p, self.param_value)
        else:
            raise ValueError(f"Unsupported mutator {self.mutator!r}")

    def _named_targets(self, model: nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
        out: list[tuple[str, torch.nn.Parameter]] = []
        for name, p in model.named_parameters():
            lname = name.lower()
            if not p.is_floating_point():
                continue
            if self.target_layers and not _matches_layer(lname, self.target_layers):
                continue
            if self.operator_id in {"NSG", "NZG"} and not lname.endswith("weight"):
                continue
            if self.operator_id == "NSB" and not lname.endswith("bias"):
                continue
            if self.operator_id in {"QZQ", "QZK", "QZV"} and not lname.endswith("weight"):
                continue
            if any(pattern in lname for pattern in self.patterns):
                out.append((name, p))
        if not out and self.operator_id == "ESS":
            out = [(n, p) for n, p in model.named_parameters()
                   if p.is_floating_point() and "embedding" in n.lower()]
        return out


class _RequiresGradFault(FaultInjector):
    operator_id: str
    target_layers: tuple[int, ...]

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._backups: list[tuple[torch.nn.Parameter, bool]] = []

    def _apply(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            lname = name.lower()
            if self.target_layers and not _matches_layer(lname, self.target_layers):
                continue
            if any(s in lname for s in ("query", "key", "value", "q_proj", "k_proj",
                                        "v_proj", "q_lin", "k_lin", "v_lin")):
                self._backups.append((p, p.requires_grad))
                p.requires_grad_(False)

    def _undo(self, model: nn.Module) -> None:
        for p, flag in self._backups:
            p.requires_grad_(flag)
        self._backups.clear()


class _ActivationFault(FaultInjector):
    param_value: Any | None

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._backups: list[tuple[nn.Module, str, Any]] = []

    def _apply(self, model: nn.Module) -> None:
        replacement = _activation_module(self.param_value)
        for module in model.modules():
            for attr in ("activation", "act", "intermediate_act_fn"):
                if hasattr(module, attr):
                    self._backups.append((module, attr, getattr(module, attr)))
                    setattr(module, attr, replacement)

    def _undo(self, model: nn.Module) -> None:
        for module, attr, original in self._backups:
            setattr(module, attr, original)
        self._backups.clear()


class _LayerNormEpsilonFault(FaultInjector):
    param_value: Any | None

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._backups: list[tuple[nn.Module, float]] = []

    def _apply(self, model: nn.Module) -> None:
        value = float(self.param_value if self.param_value is not None else 1e-4)
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) or hasattr(module, "eps"):
                eps = getattr(module, "eps", None)
                if isinstance(eps, (float, int)):
                    self._backups.append((module, float(eps)))
                    module.eps = value

    def _undo(self, model: nn.Module) -> None:
        for module, eps in self._backups:
            module.eps = eps
        self._backups.clear()


class _ModelAttributeFault(FaultInjector):
    operator_id: str
    param_value: Any | None
    severity: str | None

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._had_attr = False
        self._original: Any = None

    def _apply(self, model: nn.Module) -> None:
        attr = f"_defaultplusplus_{self.operator_id.lower()}"
        self._attr = attr
        self._had_attr = hasattr(model, attr)
        self._original = getattr(model, attr, None)
        setattr(model, attr, self.param_value if self.param_value is not None else self.severity)

    def _undo(self, model: nn.Module) -> None:
        if self._had_attr:
            setattr(model, self._attr, self._original)
        elif hasattr(model, self._attr):
            delattr(model, self._attr)


class _ForwardFault(DynamicFault):
    operator_id: str
    patterns: tuple[str, ...]
    target_layers: tuple[int, ...]
    param_value: Any | None

    def __init__(self, model: nn.Module):
        super().__init__(model)
        # CLK stash: holds the most recent past_key_values seen on a
        # clean forward so a future cache-less forward leaks it back in.
        self._cache_stash: Any = None

    def target_modules(self, model: nn.Module) -> list[nn.Module]:
        matches = []
        for name, module in model.named_modules():
            lname = name.lower()
            if not lname:
                continue
            if self.target_layers and not _matches_layer(lname, self.target_layers):
                continue
            if any(pattern in lname for pattern in self.patterns):
                matches.append(module)
        return matches or [model]

    def make_faulty_forward(self,
                            module: nn.Module,
                            original_forward: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            args2, kwargs2 = self._mutate_inputs(args, kwargs)
            out = original_forward(*args2, **kwargs2)
            return self._mutate_output(out)
        return _wrapped

    def _mutate_inputs(self,
                       args: tuple[Any, ...],
                       kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        out_kwargs = dict(kwargs)
        for key, value in list(out_kwargs.items()):
            lkey = key.lower()
            if "mask" in lkey and torch.is_tensor(value):
                out_kwargs[key] = _mutate_mask(self.operator_id, value, self.param_value)
            elif (self.operator_id in {"PSI", "PTL"}
                  and "position" in lkey and torch.is_tensor(value)):
                out_kwargs[key] = _mutate_positions(self.operator_id, value, self.param_value)
            elif self.operator_id == "KMD" and "dropout" in lkey:
                out_kwargs[key] = float(self.param_value if self.param_value is not None else 0.1)

        if self.operator_id in {"CST", "CDU", "COB", "CTR", "CLK"}:
            out_kwargs = self._mutate_cache_inputs(out_kwargs)
        return args, out_kwargs

    def _mutate_cache_inputs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply CST/CDU/COB/CTR/CLK to the model-forward kwargs.

        CTR / COB / CST / CDU mutate ``past_key_values`` when present.
        CLK injects the previously-stashed cache when the caller did
        *not* pass one, simulating cross-request reuse.
        """
        op = self.operator_id
        cache_key = _find_cache_key(kwargs)

        if op == "CLK":
            # If the caller did not supply a cache and we have a stash,
            # inject the stash. The forward then runs as if the user had
            # passed an existing cache. For a fresh request, this is the
            # cross-request leak the operator describes.
            if cache_key is None and self._cache_stash is not None:
                kwargs["past_key_values"] = self._cache_stash
                kwargs["use_cache"] = True
                # Force position_ids to the leaked offset so HF treats
                # the new tokens as continuation rather than a restart.
                offset = _cache_seq_length(self._cache_stash)
                input_ids = kwargs.get("input_ids")
                if offset and torch.is_tensor(input_ids) and input_ids.dim() == 2:
                    seq = input_ids.size(1)
                    kwargs["position_ids"] = torch.arange(
                        offset, offset + seq, device=input_ids.device, dtype=torch.long
                    ).unsqueeze(0).expand(input_ids.size(0), -1)
            return kwargs

        if cache_key is None:
            return kwargs
        cache = kwargs[cache_key]
        if cache is None:
            return kwargs

        if op == "CTR":
            length = int(self.param_value) if self.param_value is not None else 8
            kwargs[cache_key] = _truncate_cache(cache, length)
        elif op == "CDU":
            # Desynchronized update: drop the most recent appended K, V so
            # attention and prediction read cached states that are one step
            # behind the current decoding step. Cache tensor shape is
            # otherwise preserved per layer.
            kwargs[cache_key] = _desync_cache(cache)
        elif op == "COB":
            shift = int(self.param_value) if self.param_value is not None else 1
            kwargs[cache_key] = _shift_cache(cache, shift)
        elif op == "CST":
            scope = str(self.param_value) if self.param_value is not None else "all"
            kwargs[cache_key] = _stale_cache(cache, scope)
        return kwargs

    def _mutate_output(self, out: Any) -> Any:
        op = self.operator_id
        if op in {"OSL", "RSR"}:
            return _map_tensors(out, lambda x: x * float(self.param_value or 0.5), first_only=True)
        if op in {"RRS", "POE", "VSH"}:
            return _map_tensors(out, torch.zeros_like, first_only=True)
        if op == "RIN":
            sigma = float(self.param_value if self.param_value is not None else 0.01)
            return _map_tensors(out, lambda x: x + torch.randn_like(x) * sigma, first_only=True)
        if op == "SPD":
            p = float(self.param_value if self.param_value is not None else 0.1)
            return _map_tensors(out, lambda x: torch.nn.functional.dropout(x, p=p, training=True),
                                first_only=True)
        if op == "SUC":
            return _map_tensors(out, lambda x: x.to(torch.float16).to(x.dtype), first_only=True)
        if op == "KRP":
            low = _reduced_precision_dtype(self.param_value)
            return _map_tensors(out, lambda x: x.to(low).to(x.dtype), first_only=True)
        if op == "OOD":
            return _map_tensors(out, _rotate_last_dim, first_only=True)
        if op == "CLK":
            # Stash this run's cache so a future request can read it.
            # We keep the reference (no deep copy) — the leak is to
            # the live cache object exactly as in a real cross-request
            # bug where two requests share state.
            past_kv = getattr(out, "past_key_values", None)
            if past_kv is not None:
                self._cache_stash = past_kv
        return out


def _matches_layer(name: str, layers: Sequence[int]) -> bool:
    zero_based = {int(i) - 1 for i in layers if int(i) > 0}
    one_based = {int(i) for i in layers}
    tokens = name.replace("_", ".").split(".")
    ints = {int(tok) for tok in tokens if tok.isdigit()}
    return bool(ints & zero_based or ints & one_based)


def _fraction(value: Any | None) -> float:
    if value is None:
        return 0.1
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0)


def _n_rows(param: torch.Tensor, frac: float) -> int:
    rows = param.shape[0] if param.ndim > 0 else 1
    return max(1, min(rows, int(round(rows * frac))))


def _zero_rows(param: torch.Tensor, frac: float) -> None:
    if param.ndim == 0:
        param.zero_()
        return
    param[:_n_rows(param, frac)].zero_()


def _zero_cols(param: torch.Tensor, frac: float) -> None:
    """Zero a contiguous block of feature columns (last dim), preserving shape."""
    if param.ndim < 2:
        _zero_rows(param, frac)
        return
    cols = param.shape[-1]
    n = max(1, min(cols, int(round(cols * frac))))
    param[..., :n].zero_()


def _swap_rows(param: torch.Tensor, frac: float) -> None:
    if param.ndim == 0 or param.shape[0] < 2:
        return
    n = min(_n_rows(param, frac), param.shape[0] // 2)
    a = param[:n].clone()
    param[:n].copy_(param[n:2 * n])
    param[n:2 * n].copy_(a)


_QSW_Q_TOKENS = ("query", "q_proj", "q_lin")
_QSW_K_TOKENS = ("key", "k_proj", "k_lin")


def _qsw_classify(name: str) -> str | None:
    """Return 'q' or 'k' if ``name`` belongs to a Q/K projection, else None."""
    parts = name.lower().replace("/", ".").split(".")
    for token in parts:
        if token in _QSW_Q_TOKENS:
            return "q"
        if token in _QSW_K_TOKENS:
            return "k"
    return None


def _qsw_block_key(name: str) -> str:
    """Strip the Q/K leaf so query and key in the same block share a key."""
    parts = name.split(".")
    pruned: list[str] = []
    for part in parts:
        low = part.lower()
        if low in _QSW_Q_TOKENS or low in _QSW_K_TOKENS:
            break
        pruned.append(part)
    return ".".join(pruned)


def _qsw_kind(name: str) -> str:
    """Return 'weight' or 'bias' (or the raw leaf) so we pair like with like."""
    return name.rsplit(".", 1)[-1].lower()


def _swap_qk_within_blocks(targets: list[tuple[str, torch.nn.Parameter]]) -> int:
    """Swap each (Q, K) parameter pair sharing an attention-block prefix.

    Pairs only across the same block and the same parameter kind
    (``weight`` with ``weight``, ``bias`` with ``bias``) and only when
    shapes match. Returns the number of swaps performed.
    """
    groups: dict[tuple[str, str], dict[str, torch.nn.Parameter]] = {}
    for name, param in targets:
        role = _qsw_classify(name)
        if role is None:
            continue
        key = (_qsw_block_key(name), _qsw_kind(name))
        groups.setdefault(key, {})[role] = param

    swaps = 0
    for slot in groups.values():
        q = slot.get("q")
        k = slot.get("k")
        if q is None or k is None or q.shape != k.shape:
            continue
        tmp = q.detach().clone()
        q.copy_(k)
        k.copy_(tmp)
        swaps += 1
    return swaps


def _tie_rows(param: torch.Tensor) -> None:
    if param.ndim == 0 or param.shape[0] < 2:
        return
    param[1:].copy_(param[:1].expand_as(param[1:]))


def _initialize(param: torch.Tensor, scheme: Any | None) -> None:
    name = str(scheme or "zeros").lower()
    if name in {"zeros", "zero"}:
        param.zero_()
    elif name in {"constant_one", "ones", "one"}:
        param.fill_(1.0)
    elif name == "kaiming" and param.ndim >= 2:
        nn.init.kaiming_uniform_(param)
    elif name == "xavier" and param.ndim >= 2:
        nn.init.xavier_uniform_(param)
    else:
        nn.init.uniform_(param, -0.02, 0.02)


def _reduced_precision_dtype(name: Any | None) -> torch.dtype:
    key = str(name or "fp16").lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"tf32", "fp16", "float16", "half"}:
        # tf32 has no distinct tensor dtype; fp16 is the closest
        # reduced-precision round-trip we can apply to a trace tensor.
        return torch.float16
    return torch.float16


def _activation_module(name: Any | None) -> nn.Module:
    key = str(name or "relu").lower()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


def _mutate_mask(op: str, mask: torch.Tensor, value: Any | None) -> torch.Tensor:
    if op == "MZM":
        return torch.zeros_like(mask)
    if op == "MIM":
        if mask.dtype == torch.bool:
            return ~mask
        return torch.where(mask == 0, torch.ones_like(mask), torch.zeros_like(mask))
    if op == "MRM" and mask.ndim >= 2:
        return mask.transpose(-1, -2).contiguous()
    if op in {"MCB", "VEC"} and mask.ndim >= 2:
        out = mask.clone()
        n = out.shape[-1]
        visible = max(1, int(round(n * _fraction(value))))
        out[..., :visible] = 0 if out.dtype != torch.bool else True
        return out
    return mask


def _mutate_positions(op: str, positions: torch.Tensor, value: Any | None) -> torch.Tensor:
    if op == "PSI":
        return positions + int(value if value is not None else 1)
    if op == "PTL":
        cutoff = int(value if value is not None else 64)
        return torch.clamp(positions, max=max(cutoff - 1, 0))
    return positions


def _map_tensors(obj: Any,
                 fn: Callable[[torch.Tensor], torch.Tensor],
                 *,
                 first_only: bool) -> Any:
    done = False

    def apply(x: Any) -> Any:
        nonlocal done
        if torch.is_tensor(x) and (not first_only or not done):
            done = True
            return fn(x)
        if isinstance(x, tuple):
            return tuple(apply(v) for v in x)
        if isinstance(x, list):
            return [apply(v) for v in x]
        if hasattr(x, "logits") and torch.is_tensor(x.logits) and (not first_only or not done):
            done = True
            try:
                x.logits = fn(x.logits)
            except Exception:
                return x
        return x

    return apply(obj)


def _rotate_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0 or x.shape[-1] < 2:
        return x
    return torch.roll(x, shifts=1, dims=-1)


# ── Cache mutation helpers (CST / COB / CTR / CLK) ───────────────────────
_CACHE_KWARG_NAMES = ("past_key_values", "past_key_value", "layer_past",
                      "past_kv", "kv_cache")


def _find_cache_key(kwargs: dict[str, Any]) -> str | None:
    for name in _CACHE_KWARG_NAMES:
        if name in kwargs and kwargs[name] is not None:
            return name
    return None


def _iter_cache_layers(cache: Any):
    """Yield ``(idx, key_tensor, value_tensor)`` triples for each layer.

    Supports three HF cache shapes:
      * Newer ``DynamicCache`` with per-layer ``DynamicLayer`` objects
        exposing ``.keys`` / ``.values`` under ``cache.layers``.
      * Older ``DynamicCache`` exposing parallel ``key_cache`` /
        ``value_cache`` lists.
      * Legacy tuple-of-tuples ``((k0, v0), (k1, v1), ...)``.
    """
    if hasattr(cache, "layers") and hasattr(cache.layers, "__iter__"):
        for i, layer in enumerate(cache.layers):
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if torch.is_tensor(k) and torch.is_tensor(v):
                yield i, k, v
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = list(cache.key_cache)
        values = list(cache.value_cache)
        for i, (k, v) in enumerate(zip(keys, values)):
            yield i, k, v
        return
    if isinstance(cache, (list, tuple)) and cache:
        first = cache[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2 \
                and isinstance(first[0], torch.Tensor):
            for i, layer in enumerate(cache):
                yield i, layer[0], layer[1]


def _set_cache_layer(cache: Any, idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    """In-place replace layer ``idx``'s K/V tensors inside ``cache``."""
    if hasattr(cache, "layers") and hasattr(cache.layers, "__iter__"):
        try:
            layer = cache.layers[idx]
        except (IndexError, KeyError, TypeError):
            return
        try:
            layer.keys = k
            layer.values = v
        except AttributeError:
            pass
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if idx < len(cache.key_cache):
            cache.key_cache[idx] = k
        if idx < len(cache.value_cache):
            cache.value_cache[idx] = v
        return
    if isinstance(cache, list) and idx < len(cache):
        cache[idx] = (k, v)
    # Tuples are immutable; the caller rebuilds the cache via _rebuild_cache.


def _rebuild_cache(cache: Any, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Return a cache object of the same shape as ``cache`` with new layers."""
    # DynamicCache (either flavor): mutated in place via _set_cache_layer.
    if hasattr(cache, "layers") and hasattr(cache.layers, "__iter__"):
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache
    if isinstance(cache, tuple):
        return tuple(layers)
    if isinstance(cache, list):
        return list(layers)
    return cache


def _cache_seq_length(cache: Any) -> int:
    """Return the K/V sequence length of the first usable layer (0 if empty)."""
    for _idx, k, _v in _iter_cache_layers(cache):
        if torch.is_tensor(k) and k.dim() >= 3:
            return int(k.shape[-2])
    return 0


def _clone_cache(cache: Any) -> Any:
    """Deep-copy a cache object's tensors so later forwards can't share state.

    The newer ``DynamicCache.layers`` shape is the hardest to rebuild
    cleanly because per-layer ``DynamicLayer`` objects do book-keeping
    inside ``__init__``; we let HF reconstruct one via
    ``DynamicCache.from_legacy_cache`` when available, falling back to
    a tuple-of-tuples representation that any HF ≥ 4.40 forward also
    accepts.
    """
    new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _, k, v in _iter_cache_layers(cache):
        new_layers.append((
            k.detach().clone() if torch.is_tensor(k) else k,
            v.detach().clone() if torch.is_tensor(v) else v,
        ))

    # New DynamicCache (.layers): rebuild via the legacy adapter so the
    # internal DynamicLayer state is consistent.
    if hasattr(cache, "layers") and hasattr(cache.layers, "__iter__"):
        cls = type(cache)
        legacy = tuple(new_layers)
        for ctor in ("from_legacy_cache",):
            from_legacy = getattr(cls, ctor, None)
            if callable(from_legacy):
                try:
                    return from_legacy(legacy)
                except Exception:
                    break
        # Fall through to legacy tuple — HF still accepts it.
        return legacy

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            clone = type(cache)()
        except TypeError:
            clone = cache.__class__.__new__(cache.__class__)
            clone.__init__()
        clone.key_cache = [k for k, _ in new_layers]
        clone.value_cache = [v for _, v in new_layers]
        if hasattr(cache, "_seen_tokens"):
            clone._seen_tokens = getattr(cache, "_seen_tokens")
        return clone
    if isinstance(cache, tuple):
        return tuple(new_layers)
    if isinstance(cache, list):
        return list(new_layers)
    return cache


def _truncate_cache(cache: Any, length: int) -> Any:
    """CTR: keep only the last ``length`` cached positions per layer."""
    length = max(0, int(length))
    new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, k, v in _iter_cache_layers(cache):
        if torch.is_tensor(k) and torch.is_tensor(v) and k.dim() >= 3:
            seq = k.shape[-2]
            keep = min(length, seq)
            k_new = k[..., -keep:, :] if keep > 0 else k[..., :0, :]
            v_new = v[..., -keep:, :] if keep > 0 else v[..., :0, :]
        else:
            k_new, v_new = k, v
        _set_cache_layer(cache, idx, k_new, v_new)
        new_layers.append((k_new, v_new))
    return _rebuild_cache(cache, new_layers)


def _shift_cache(cache: Any, shift: int) -> Any:
    """COB: misalign K/V along the sequence axis by ``shift`` positions.

    A negative ``shift`` drops the leading positions and pads the tail
    with zeros so subsequent reads land one past the true position.
    A positive ``shift`` duplicates the leading row to push the rest
    of the cache forward.
    """
    shift = int(shift)
    if shift == 0:
        return cache
    new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, k, v in _iter_cache_layers(cache):
        if torch.is_tensor(k) and torch.is_tensor(v) and k.dim() >= 3:
            k_new = _shift_along_seq(k, shift)
            v_new = _shift_along_seq(v, shift)
        else:
            k_new, v_new = k, v
        _set_cache_layer(cache, idx, k_new, v_new)
        new_layers.append((k_new, v_new))
    return _rebuild_cache(cache, new_layers)


def _shift_along_seq(t: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift a (..., seq, head_dim) tensor by ``shift`` along the seq axis.

    Positions vacated by the shift are filled with zeros. The output
    keeps the same shape as the input.
    """
    if shift == 0 or t.shape[-2] == 0:
        return t.clone()
    seq = t.shape[-2]
    out = torch.zeros_like(t)
    if shift > 0:
        # Push contents forward: out[..., shift:, :] <- t[..., :seq-shift, :]
        keep = max(0, seq - shift)
        if keep > 0:
            out[..., shift:shift + keep, :] = t[..., :keep, :]
    else:
        # shift < 0: pull contents back; out[..., :seq+shift, :] <- t[..., -shift:, :]
        s = -shift
        keep = max(0, seq - s)
        if keep > 0:
            out[..., :keep, :] = t[..., s:s + keep, :]
    return out


def _desync_cache(cache: Any) -> Any:
    """CDU: withhold the most recent appended K/V from every cache layer.

    Dropping the last cached position across all layers leaves attention
    and prediction reading states that lag the current decoding step by
    one position, while each per-layer cache tensor keeps a valid shape.
    """
    new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, k, v in _iter_cache_layers(cache):
        if torch.is_tensor(k) and torch.is_tensor(v) and k.dim() >= 3 and k.shape[-2] >= 2:
            k_new = k[..., :-1, :]
            v_new = v[..., :-1, :]
        else:
            k_new, v_new = k, v
        _set_cache_layer(cache, idx, k_new, v_new)
        new_layers.append((k_new, v_new))
    return _rebuild_cache(cache, new_layers)


def _cst_layer_indices(scope: str, n_layers: int) -> set[int]:
    """Pick which layer indices CST should staleify."""
    s = (scope or "all").strip().lower()
    if n_layers <= 0:
        return set()
    if s == "all":
        return set(range(n_layers))
    if s == "first":
        return {0}
    if s == "last":
        return {n_layers - 1}
    if s in ("middle", "mid"):
        return {n_layers // 2}
    return set(range(n_layers))


def _stale_cache(cache: Any, scope: str) -> Any:
    """CST: replace selected layers' K/V with a 1-step-stale snapshot.

    A "1-step-stale" snapshot for a cache of length ``T`` is the cache
    of length ``T-1`` — i.e. drop the most recent appended position.
    The model then attends as if the most recent token had not been
    written into the cache yet.
    """
    layer_pairs = list(_iter_cache_layers(cache))
    n_layers = len(layer_pairs)
    targets = _cst_layer_indices(scope, n_layers)

    new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, k, v in layer_pairs:
        if idx in targets and torch.is_tensor(k) and torch.is_tensor(v) and k.dim() >= 3:
            seq = k.shape[-2]
            if seq >= 2:
                k_new = k[..., :-1, :]
                v_new = v[..., :-1, :]
            else:
                # No prior step to revert to; zero the slot so the
                # operator still produces an observable change.
                k_new = torch.zeros_like(k)
                v_new = torch.zeros_like(v)
        else:
            k_new, v_new = k, v
        _set_cache_layer(cache, idx, k_new, v_new)
        new_layers.append((k_new, v_new))
    return _rebuild_cache(cache, new_layers)
