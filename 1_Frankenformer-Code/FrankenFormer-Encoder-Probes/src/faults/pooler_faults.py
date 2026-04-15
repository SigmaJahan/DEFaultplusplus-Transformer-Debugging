import torch
import torch.nn as nn
from .base_fault import BaseFault


def _get_pooler(model):
    for attr in ('pooler', 'bert.pooler', 'roberta.pooler', 'electra.pooler'):
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            if obj is not None:
                return obj
        except AttributeError:
            continue
    return None


class PoolerScaleFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        alpha = params.get('alpha', 0.0)
        pooler = _get_pooler(model)
        if pooler is None or not hasattr(pooler, 'dense'):
            self._active = False
            return
        self._backup_param('weight', pooler.dense.weight.data.clone())
        pooler.dense.weight.data.mul_(alpha)
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        pooler = _get_pooler(model)
        pooler.dense.weight.data.copy_(self._backups['weight'])
        self._active = False


class PoolerZeroFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        pooler = _get_pooler(model)
        if pooler is None:
            self._active = False
            return
        self._orig_forward = pooler.forward

        def patched_forward(hidden_states):
            result = self._orig_forward(hidden_states)
            return torch.zeros_like(result)

        self._backup_param('forward', self._orig_forward)
        pooler.forward = patched_forward
        self._pooler_ref = pooler
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._pooler_ref.forward = self._backups['forward']
        self._active = False


class PoolerNoiseFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        std = params.get('std', 0.1)
        pooler = _get_pooler(model)
        if pooler is None:
            self._active = False
            return
        self._orig_forward = pooler.forward

        def patched_forward(hidden_states):
            result = self._orig_forward(hidden_states)
            return result + torch.randn_like(result) * std

        self._backup_param('forward', self._orig_forward)
        pooler.forward = patched_forward
        self._pooler_ref = pooler
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._pooler_ref.forward = self._backups['forward']
        self._active = False


POOLER_FAULTS = {
    'pooler_scale': PoolerScaleFault,
    'pooler_zero': PoolerZeroFault,
    'pooler_noise': PoolerNoiseFault,
}


def create_pooler_fault(name: str, **kwargs) -> BaseFault:
    return POOLER_FAULTS[name](**kwargs)
