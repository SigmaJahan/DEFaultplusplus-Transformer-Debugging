"""
Embedding Faults (Group 5)

Implements:
- EmbeddingZeroFault: zero/corrupt a subset of token embeddings
- EmbeddingSwapFault: swap embeddings between token IDs
- TypeEmbeddingDropFault: remove or scale type/segment embeddings
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from src.faults.base_fault import BaseFault


class _EmbeddingFault(BaseFault):
    """Helper base to access embeddings safely."""

    def _get_embeddings(self):
        return self.get_embeddings_module()

    def _get_embedding_weight(self):
        """
        Get the embedding weight tensor for both encoder and decoder models.

        For encoder models (BERT, RoBERTa): emb_module.word_embeddings.weight
        For decoder models (GPT-2, DistilGPT2): emb_module.weight (direct nn.Embedding)

        Returns:
            The embedding weight tensor
        """
        emb_module = self._get_embeddings()

        # Check if this is a direct nn.Embedding (decoder) or wrapped embeddings (encoder)
        if isinstance(emb_module, nn.Embedding):
            # Direct embedding layer (decoder models like GPT-2)
            return emb_module.weight
        elif hasattr(emb_module, 'word_embeddings'):
            # Wrapped embeddings (encoder models like BERT)
            return emb_module.word_embeddings.weight
        else:
            raise ValueError(
                f"Cannot access embedding weights from {type(emb_module)}. "
                f"Expected nn.Embedding or module with 'word_embeddings' attribute."
            )


class EmbeddingZeroFault(_EmbeddingFault):
    """Zero out or corrupt a subset of token embeddings."""

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int = 0,
        fraction: float = 0.05,
        noise_std: float = 0.0
    ):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="embedding_zero",
            description=f"Zero/corrupt {fraction:.2f} of token embeddings"
        )
        self.fraction = float(fraction)
        self.noise_std = float(noise_std)
        self.indices: Optional[torch.Tensor] = None

    def inject(self) -> None:
        if self.is_injected:
            return

        weight = self._get_embedding_weight()
        vocab = weight.size(0)
        count = max(1, int(vocab * self.fraction))

        gen = torch.Generator()
        gen.manual_seed(int(self.fraction * 10000))
        perm = torch.randperm(vocab, generator=gen)
        indices = perm[:count]
        self.indices = indices

        self.original_state = {
            "embeddings": weight.detach().clone()
        }

        with torch.no_grad():
            if self.noise_std > 0:
                noise = torch.randn_like(weight[indices]) * self.noise_std
                weight[indices] = noise
            else:
                weight[indices] = 0

        self.is_injected = True
        self._update_fault_metadata(
            "embedding_zero",
            {"indices": indices.tolist(), "fraction": self.fraction, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} on {len(indices)} tokens")

    def restore(self) -> None:
        if not self.is_injected:
            return
        try:
            weight = self._get_embedding_weight()
            with torch.no_grad():
                weight.copy_(self.original_state["embeddings"])
        except Exception:
            pass
        self.original_state = {}
        self.indices = None
        self.is_injected = False
        self._update_fault_metadata("embedding_zero", None)
        self._log_info(f"Restored from {self.fault_name}")


class EmbeddingSwapFault(_EmbeddingFault):
    """Swap embeddings between token IDs."""

    def __init__(self, model: nn.Module, layer_idx: int = 0, swaps: int = 1):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="embedding_swap",
            description=f"Swap {swaps} embedding pairs"
        )
        self.swaps = max(1, int(swaps))
        self.pairs: List[List[int]] = []

    def inject(self) -> None:
        if self.is_injected:
            return

        weight = self._get_embedding_weight()
        vocab = weight.size(0)

        gen = torch.Generator()
        gen.manual_seed(1337 + self.swaps)
        perm = torch.randperm(vocab, generator=gen)
        # Ensure even length to avoid view errors when vocab size is odd
        even_count = (vocab // 2) * 2
        perm = perm[:even_count]
        pairs = perm.view(-1, 2)[:self.swaps]
        self.pairs = pairs.tolist()

        self.original_state = {
            "embeddings": weight.detach().clone()
        }

        with torch.no_grad():
            for pair in pairs:
                a, b = int(pair[0].item()), int(pair[1].item())
                tmp = weight[a].clone()
                weight[a] = weight[b]
                weight[b] = tmp

        self.is_injected = True
        self._update_fault_metadata(
            "embedding_swap",
            {"pairs": self.pairs, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} with {len(self.pairs)} swaps")

    def restore(self) -> None:
        if not self.is_injected:
            return
        try:
            weight = self._get_embedding_weight()
            with torch.no_grad():
                weight.copy_(self.original_state["embeddings"])
        except Exception:
            pass
        self.original_state = {}
        self.pairs = []
        self.is_injected = False
        self._update_fault_metadata("embedding_swap", None)
        self._log_info(f"Restored from {self.fault_name}")


class TypeEmbeddingDropFault(_EmbeddingFault):
    """Drop or scale type/segment embeddings."""

    def __init__(self, model: nn.Module, layer_idx: int = 0, scale: float = 0.0):
        super().__init__(
            model=model,
            layer_idx=layer_idx,
            fault_name="type_embedding_drop",
            description=f"Scale type embeddings by {scale}"
        )
        self.scale = float(scale)
        self.original_state = {}

    def inject(self) -> None:
        if self.is_injected:
            return
        emb_module = self._get_embeddings()
        token_type_embeddings = getattr(emb_module, 'token_type_embeddings', None)
        if token_type_embeddings is None:
            # Wrap forward to mark injection, even though no token-type embeddings exist
            self.original_state = {"forward": emb_module.forward}
            original_forward = emb_module.forward

            def passthrough_forward(*args, **kwargs):
                return original_forward(*args, **kwargs)

            emb_module.forward = passthrough_forward
            self.is_injected = True
            self._update_fault_metadata(
                "type_embedding_drop",
                {"scale": self.scale, "note": "no_token_type_embeddings_present", "layer": self.layer_idx}
            )
            self._log_warning("type_embedding_drop applied as no-op (token_type_embeddings missing)")
            return

        self.original_state = {
            "token_type_weight": token_type_embeddings.weight.detach().clone()
        }
        with torch.no_grad():
            token_type_embeddings.weight.mul_(self.scale)

        self.is_injected = True
        self._update_fault_metadata(
            "type_embedding_drop",
            {"scale": self.scale, "layer": self.layer_idx}
        )
        self._log_info(f"Injected {self.fault_name} (scale={self.scale})")

    def restore(self) -> None:
        if not self.is_injected:
            return
        try:
            emb_module = self._get_embeddings()
            token_type_embeddings = getattr(emb_module, 'token_type_embeddings', None)
            if token_type_embeddings is not None and "token_type_weight" in self.original_state:
                with torch.no_grad():
                    token_type_embeddings.weight.copy_(self.original_state["token_type_weight"])
            if "forward" in self.original_state:
                emb_module.forward = self.original_state["forward"]
        except Exception:
            pass
        self.original_state = {}
        self.is_injected = False
        self._update_fault_metadata("type_embedding_drop", None)
        self._log_info(f"Restored from {self.fault_name}")


EMBEDDING_FAULTS: Dict[str, Any] = {
    "embedding_zero": EmbeddingZeroFault,
    "embedding_swap": EmbeddingSwapFault,
    "type_embedding_drop": TypeEmbeddingDropFault,
}


def create_embedding_fault(
    fault_type: str,
    model: nn.Module,
    layer_idx: int = 0,
    **kwargs
) -> BaseFault:
    """Factory for embedding faults."""
    if fault_type not in EMBEDDING_FAULTS:
        raise ValueError(f"Unknown embedding fault type: {fault_type}")
    return EMBEDDING_FAULTS[fault_type](model, layer_idx, **kwargs)
