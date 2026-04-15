import torch
import torch.nn as nn
from .base_fault import BaseFault


def _get_classifier(model):
    for attr in ('classifier', 'cls', 'qa_outputs', 'pre_classifier'):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Linear):
                return mod
            if hasattr(mod, 'dense'):
                return mod.dense
            if hasattr(mod, 'decoder'):
                return mod.decoder
    raise ValueError("Cannot find classifier/output head")


class OutScaleFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        scale = params.get('scale', 10.0)
        clf = _get_classifier(model)
        self._backup_param('weight', clf.weight.data.clone())
        clf.weight.data.mul_(scale)
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        clf = _get_classifier(model)
        clf.weight.data.copy_(self._backups['weight'])
        self._active = False


class OutRowDropFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        rows = params.get('rows', [0])
        clf = _get_classifier(model)
        self._backup_param('weight', clf.weight.data.clone())
        self._backup_param('bias', clf.bias.data.clone() if clf.bias is not None else None)
        for r in rows:
            if r < clf.weight.shape[0]:
                clf.weight.data[r] = 0
                if clf.bias is not None:
                    clf.bias.data[r] = 0
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        clf = _get_classifier(model)
        clf.weight.data.copy_(self._backups['weight'])
        if self._backups['bias'] is not None:
            clf.bias.data.copy_(self._backups['bias'])
        self._active = False


class OutNoiseFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        std = params.get('std', 0.1)
        clf = _get_classifier(model)
        self._backup_param('weight', clf.weight.data.clone())
        clf.weight.data.add_(torch.randn_like(clf.weight.data) * std)
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        clf = _get_classifier(model)
        clf.weight.data.copy_(self._backups['weight'])
        self._active = False


OUTPUT_FAULTS = {
    'out_scale': OutScaleFault,
    'out_row_drop': OutRowDropFault,
    'out_noise': OutNoiseFault,
}


def create_output_fault(name: str, **kwargs) -> BaseFault:
    return OUTPUT_FAULTS[name](**kwargs)
