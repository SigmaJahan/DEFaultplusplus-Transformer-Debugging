import torch.nn as nn
from .base_fault import BaseFault


def _get_layernorm(model, layer_idx):
    for attr in ('encoder.layer', 'bert.encoder.layer', 'roberta.encoder.layer',
                 'electra.encoder.layer', 'transformer.layer', 'distilbert.transformer.layer'):
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            layer = obj[layer_idx]
            break
        except AttributeError:
            continue
    else:
        raise ValueError("Cannot find encoder layers")
    if hasattr(layer, 'attention') and hasattr(layer.attention, 'output'):
        return layer.attention.output.LayerNorm
    if hasattr(layer, 'sa_layer_norm'):
        return layer.sa_layer_norm
    if hasattr(layer, 'output') and hasattr(layer.output, 'LayerNorm'):
        return layer.output.LayerNorm
    raise ValueError("Cannot find LayerNorm in layer")


class LNGammaFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        scale = params.get('scale', 0.0)
        ln = _get_layernorm(model, layer_idx)
        self._backup_param('weight', ln.weight.data.clone())
        ln.weight.data.mul_(scale)
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        ln = _get_layernorm(model, self._model_ref[1])
        ln.weight.data.copy_(self._backups['weight'])
        self._active = False


class LNBetaFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        shift = params.get('shift', 100.0)
        ln = _get_layernorm(model, layer_idx)
        self._backup_param('bias', ln.bias.data.clone())
        ln.bias.data.add_(shift)
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        ln = _get_layernorm(model, self._model_ref[1])
        ln.bias.data.copy_(self._backups['bias'])
        self._active = False


class LNStatsFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        epsilon = params.get('epsilon', 1.0)
        ln = _get_layernorm(model, layer_idx)
        self._backup_param('eps', ln.eps)
        ln.eps = epsilon
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        ln = _get_layernorm(model, self._model_ref[1])
        ln.eps = self._backups['eps']
        self._active = False


LAYERNORM_FAULTS = {
    'ln_gamma': LNGammaFault,
    'ln_beta': LNBetaFault,
    'ln_stats': LNStatsFault,
}


def create_layernorm_fault(name: str, **kwargs) -> BaseFault:
    return LAYERNORM_FAULTS[name](**kwargs)
