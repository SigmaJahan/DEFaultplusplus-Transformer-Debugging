"""
Positional Faults (E4 Category)

Implements positional encoding faults for encoder models (DistilBERT/BERT/Roberta/ModernBERT).
"""

from typing import Optional, Callable
import torch
import torch.nn as nn

from src.faults.base_fault import BaseFault


def _emb_forward(
    embeddings: nn.Module,
    *,
    input_ids: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor],
    position_shift: int = 0,
    position_clamp: Optional[int] = None,
    drop_positions: bool = False,
    position_scale: float = 1.0,
    original_forward: Optional[Callable] = None,
) -> torch.Tensor:
    """Rebuild embeddings output with optional positional perturbations.

    Supports both encoder models (BERT-style with word_embeddings attribute)
    and decoder models (GPT-2-style with direct nn.Embedding).
    """
    if inputs_embeds is not None:
        word_embeddings = inputs_embeds
    elif input_ids is not None:
        # Handle both encoder and decoder embedding structures
        if hasattr(embeddings, 'word_embeddings'):
            # Encoder models (BERT, RoBERTa, DistilBERT)
            word_embeddings = embeddings.word_embeddings(input_ids)
        elif isinstance(embeddings, nn.Embedding):
            # Decoder models (GPT-2, DistilGPT2) - direct embedding
            # Use original_forward to avoid recursion
            if original_forward is not None:
                word_embeddings = original_forward(input_ids)
            else:
                # Fallback to __call__ on parent class (avoid recursion)
                word_embeddings = nn.Embedding.forward(embeddings, input_ids)
        else:
            raise ValueError(
                f"Cannot access word embeddings from {type(embeddings)}. "
                f"Expected nn.Embedding or module with 'word_embeddings' attribute."
            )
    else:
        raise ValueError("You must specify either input_ids or inputs_embeds")

    embeddings_output = word_embeddings

    if hasattr(embeddings, "position_embeddings") and not drop_positions:
        seq_length = word_embeddings.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=word_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand_as(word_embeddings[:, :, 0])
        if position_shift != 0:
            position_ids = position_ids + position_shift
        if position_clamp is not None:
            position_ids = torch.clamp(position_ids, max=position_clamp - 1)
        position_embeddings = embeddings.position_embeddings(position_ids)
        embeddings_output = embeddings_output + position_scale * position_embeddings

    if hasattr(embeddings, "token_type_embeddings") and token_type_ids is not None:
        embeddings_output = embeddings_output + embeddings.token_type_embeddings(token_type_ids)

    ln = getattr(embeddings, "LayerNorm", None) or getattr(embeddings, "layer_norm", None)
    if ln is not None:
        embeddings_output = ln(embeddings_output)

    if hasattr(embeddings, "dropout") and embeddings.dropout is not None:
        embeddings_output = embeddings.dropout(embeddings_output)

    return embeddings_output


class MissingPositionalFault(BaseFault):
    """
    E4.1: Missing Positional Embeddings

    Prevents positional embeddings from being added to input embeddings.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 0):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="missing_positional",
            description="Positional embeddings not added to input",
        )
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        if self.is_injected:
            return

        embeddings = self.get_embeddings_module()
        self.original_forward = embeddings.forward
        original_fwd = self.original_forward

        def faulty_forward(input_ids=None, position_ids=None, inputs_embeds=None, token_type_ids=None, **kwargs):
            return _emb_forward(
                embeddings,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids,
                drop_positions=True,
                original_forward=original_fwd,
            )

        embeddings.forward = faulty_forward
        self.is_injected = True
        self._update_fault_metadata("positional_fault", {"type": self.fault_name, "layer": self.layer_idx})
        self._log_info(f"Injected {self.fault_name}")

    def restore(self) -> None:
        if not self.is_injected:
            return
        if self.original_forward is not None:
            embeddings = self.get_embeddings_module()
            embeddings.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._log_info(f"Restored from {self.fault_name}")


class OffByOneFault(BaseFault):
    """
    E4.1: Off-by-One Positions

    Shifts position IDs by a configurable offset.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 0, shift: int = 1):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="off_by_one",
            description=f"Position IDs shifted by {shift}",
        )
        self.shift = shift
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        if self.is_injected:
            return

        embeddings = self.get_embeddings_module()
        self.original_forward = embeddings.forward
        original_fwd = self.original_forward
        shift = self.shift
        max_position = getattr(getattr(embeddings, "position_embeddings", None), "num_embeddings", None)

        def faulty_forward(input_ids=None, position_ids=None, inputs_embeds=None, token_type_ids=None, **kwargs):
            return _emb_forward(
                embeddings,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids,
                position_shift=shift,
                position_clamp=max_position,
                original_forward=original_fwd,
            )

        embeddings.forward = faulty_forward
        self.is_injected = True
        self._update_fault_metadata("positional_fault", {"type": self.fault_name, "shift": shift, "layer": self.layer_idx})
        self._log_info(f"Injected {self.fault_name} (shift={shift})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        if self.original_forward is not None:
            embeddings = self.get_embeddings_module()
            embeddings.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._log_info(f"Restored from {self.fault_name}")


class TruncatePositionsFault(BaseFault):
    """
    E4.2: Truncate Positions

    Clamps position IDs to a maximum value, causing position aliasing.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 0, max_position: int = 100):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="truncate_positions",
            description=f"Position IDs clamped to < {max_position}",
        )
        self.max_position = max_position
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        if self.is_injected:
            return

        embeddings = self.get_embeddings_module()
        self.original_forward = embeddings.forward
        original_fwd = self.original_forward
        max_position = self.max_position

        def faulty_forward(input_ids=None, position_ids=None, inputs_embeds=None, token_type_ids=None, **kwargs):
            return _emb_forward(
                embeddings,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids,
                position_clamp=max_position,
                original_forward=original_fwd,
            )

        embeddings.forward = faulty_forward
        self.is_injected = True
        self._update_fault_metadata(
            "positional_fault",
            {"type": self.fault_name, "max_position": max_position, "layer": self.layer_idx},
        )
        self._log_info(f"Injected {self.fault_name} (max_position={max_position})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        if self.original_forward is not None:
            embeddings = self.get_embeddings_module()
            embeddings.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._log_info(f"Restored from {self.fault_name}")


class DoublePositionFault(BaseFault):
    """
    E4.3: Double Position Embeddings

    Adds positional embeddings twice (or a scaled factor) to exaggerate positional information.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 0, scale: float = 2.0):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="double_position",
            description=f"Positional embeddings scaled by {scale}",
        )
        self.scale = float(scale)
        self.original_forward: Optional[Callable] = None

    def inject(self) -> None:
        if self.is_injected:
            return

        embeddings = self.get_embeddings_module()
        self.original_forward = embeddings.forward
        original_fwd = self.original_forward
        scale = self.scale

        def faulty_forward(input_ids=None, position_ids=None, inputs_embeds=None, token_type_ids=None, **kwargs):
            return _emb_forward(
                embeddings,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids,
                position_scale=scale,
                original_forward=original_fwd,
            )

        embeddings.forward = faulty_forward
        self.is_injected = True
        self._update_fault_metadata(
            "positional_fault",
            {"type": self.fault_name, "scale": scale, "layer": self.layer_idx},
        )
        self._log_info(f"Injected {self.fault_name} (scale={scale}) into layer {self.layer_idx}")

    def restore(self) -> None:
        if not self.is_injected:
            return
        if self.original_forward is not None:
            embeddings = self.get_embeddings_module()
            embeddings.forward = self.original_forward
        self.original_forward = None
        self.is_injected = False
        self._log_info(f"Restored from {self.fault_name}")


POSITIONAL_FAULTS = {
    "missing_positional": MissingPositionalFault,
    "off_by_one": OffByOneFault,
    "truncate_positions": TruncatePositionsFault,
    "double_position": DoublePositionFault,
}


def create_positional_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int = 0,
    **kwargs,
) -> BaseFault:
    """
    Factory to build positional faults with optional severity parameters.
    """
    if fault_type not in POSITIONAL_FAULTS:
        raise ValueError(
            f"Unknown positional fault '{fault_type}'. Valid options: {list(POSITIONAL_FAULTS.keys())}"
        )
    fault_cls = POSITIONAL_FAULTS[fault_type]
    return fault_cls(model, layer_idx, **kwargs)
