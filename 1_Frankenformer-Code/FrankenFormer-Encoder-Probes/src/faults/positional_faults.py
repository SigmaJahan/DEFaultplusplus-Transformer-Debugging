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


class MissingPositionalFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        emb = _get_embeddings(model)
        if hasattr(emb, 'position_embeddings'):
            self._backup_param('pos_weight', emb.position_embeddings.weight.data.clone())
            emb.position_embeddings.weight.data.zero_()
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        emb = _get_embeddings(model)
        if 'pos_weight' in self._backups:
            emb.position_embeddings.weight.data.copy_(self._backups['pos_weight'])
        self._active = False


class OffByOneFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        emb = _get_embeddings(model)
        self._orig_forward = emb.forward
        self._active = True

        def patched_forward(*args, **kwargs):
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                kwargs['position_ids'] = kwargs['position_ids'] + 1
            return self._orig_forward(*args, **kwargs)

        self._backup_param('forward', self._orig_forward)
        emb.forward = patched_forward
        self._emb_ref = emb

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._emb_ref.forward = self._backups['forward']
        self._active = False


class TruncatePositionsFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        max_pos = params.get('max_pos', 128)
        emb = _get_embeddings(model)
        if hasattr(emb, 'position_embeddings'):
            weight = emb.position_embeddings.weight.data
            self._backup_param('pos_weight', weight.clone())
            if weight.shape[0] > max_pos:
                weight[max_pos:].zero_()
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        emb = _get_embeddings(model)
        if 'pos_weight' in self._backups:
            emb.position_embeddings.weight.data.copy_(self._backups['pos_weight'])
        self._active = False


class DoublePositionFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        emb = _get_embeddings(model)
        if hasattr(emb, 'position_embeddings'):
            self._backup_param('pos_weight', emb.position_embeddings.weight.data.clone())
            emb.position_embeddings.weight.data.mul_(2.0)
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        emb = _get_embeddings(model)
        if 'pos_weight' in self._backups:
            emb.position_embeddings.weight.data.copy_(self._backups['pos_weight'])
        self._active = False


POSITIONAL_FAULTS = {
    'missing_positional': MissingPositionalFault,
    'off_by_one': OffByOneFault,
    'truncate_positions': TruncatePositionsFault,
    'double_position': DoublePositionFault,
}


def create_positional_fault(name: str, **kwargs) -> BaseFault:
    return POSITIONAL_FAULTS[name](**kwargs)
