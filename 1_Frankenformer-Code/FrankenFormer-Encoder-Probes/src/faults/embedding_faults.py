import torch
import torch.nn as nn
from .base_fault import BaseFault


def _get_embeddings(model):
    for attr in ('embeddings', 'bert.embeddings', 'roberta.embeddings',
                 'electra.embeddings', 'distilbert.embeddings'):
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("Cannot find embeddings module")


class EmbeddingZeroFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        fraction = params.get('fraction', 0.1)
        emb = _get_embeddings(model)
        weight = emb.word_embeddings.weight.data
        self._backup_param('word_weight', weight.clone())
        num_zero = int(weight.shape[0] * fraction)
        indices = torch.randperm(weight.shape[0])[:num_zero]
        weight[indices] = 0
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        emb = _get_embeddings(model)
        emb.word_embeddings.weight.data.copy_(self._backups['word_weight'])
        self._active = False


class EmbeddingSwapFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        num_swaps = params.get('num_swaps', 10)
        emb = _get_embeddings(model)
        weight = emb.word_embeddings.weight.data
        self._backup_param('word_weight', weight.clone())
        vocab_size = weight.shape[0]
        for _ in range(num_swaps):
            i, j = torch.randint(0, vocab_size, (2,)).tolist()
            weight[[i, j]] = weight[[j, i]]
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        emb = _get_embeddings(model)
        emb.word_embeddings.weight.data.copy_(self._backups['word_weight'])
        self._active = False


class TypeEmbeddingDropFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        scale = params.get('scale', 0.0)
        emb = _get_embeddings(model)
        if not hasattr(emb, 'token_type_embeddings'):
            self._active = False
            return
        self._backup_param('type_weight', emb.token_type_embeddings.weight.data.clone())
        emb.token_type_embeddings.weight.data.mul_(scale)
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        emb = _get_embeddings(model)
        emb.token_type_embeddings.weight.data.copy_(self._backups['type_weight'])
        self._active = False


EMBEDDING_FAULTS = {
    'embedding_zero': EmbeddingZeroFault,
    'embedding_swap': EmbeddingSwapFault,
    'type_embedding_drop': TypeEmbeddingDropFault,
}


def create_embedding_fault(name: str, **kwargs) -> BaseFault:
    return EMBEDDING_FAULTS[name](**kwargs)
