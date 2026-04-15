import torch
import torch.nn as nn
from .base_fault import AttentionFault
from .attention_utils import get_qkv_projections


class ZeroQueryFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        q, _, _ = get_qkv_projections(attn)
        self._backup_param('q_weight', q.weight.data.clone())
        self._backup_param('q_bias', q.bias.data.clone() if q.bias is not None else None)
        q.weight.data.zero_()
        if q.bias is not None:
            q.bias.data.zero_()
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        q, _, _ = get_qkv_projections(attn)
        q.weight.data.copy_(self._backups['q_weight'])
        if self._backups['q_bias'] is not None:
            q.bias.data.copy_(self._backups['q_bias'])
        self._active = False


class ZeroKeyFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        _, k, _ = get_qkv_projections(attn)
        self._backup_param('k_weight', k.weight.data.clone())
        self._backup_param('k_bias', k.bias.data.clone() if k.bias is not None else None)
        k.weight.data.zero_()
        if k.bias is not None:
            k.bias.data.zero_()
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        _, k, _ = get_qkv_projections(attn)
        k.weight.data.copy_(self._backups['k_weight'])
        if self._backups['k_bias'] is not None:
            k.bias.data.copy_(self._backups['k_bias'])
        self._active = False


class ZeroValueFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        _, _, v = get_qkv_projections(attn)
        self._backup_param('v_weight', v.weight.data.clone())
        self._backup_param('v_bias', v.bias.data.clone() if v.bias is not None else None)
        v.weight.data.zero_()
        if v.bias is not None:
            v.bias.data.zero_()
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        _, _, v = get_qkv_projections(attn)
        v.weight.data.copy_(self._backups['v_weight'])
        if self._backups['v_bias'] is not None:
            v.bias.data.copy_(self._backups['v_bias'])
        self._active = False


class SwappedQKFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        q, k, _ = get_qkv_projections(attn)
        self._backup_param('q_weight', q.weight.data.clone())
        self._backup_param('k_weight', k.weight.data.clone())
        q_w, k_w = k.weight.data.clone(), q.weight.data.clone()
        q.weight.data.copy_(q_w)
        k.weight.data.copy_(k_w)
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        q, k, _ = get_qkv_projections(attn)
        q.weight.data.copy_(self._backups['q_weight'])
        k.weight.data.copy_(self._backups['k_weight'])
        self._active = False


class TieHeadsFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        q, k, v = get_qkv_projections(attn)
        num_heads = getattr(attn, 'num_attention_heads', getattr(attn, 'n_heads', 1))
        head_dim = q.weight.shape[0] // num_heads
        self._backup_param('q_weight', q.weight.data.clone())
        head0 = q.weight.data[:head_dim].clone()
        for h in range(1, num_heads):
            q.weight.data[h * head_dim:(h + 1) * head_dim] = head0
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        q, _, _ = get_qkv_projections(attn)
        q.weight.data.copy_(self._backups['q_weight'])
        self._active = False


class WrongHeadDimFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        q, _, _ = get_qkv_projections(attn)
        num_heads = getattr(attn, 'num_attention_heads', getattr(attn, 'n_heads', 1))
        head_dim = q.weight.shape[0] // num_heads
        self._backup_param('head_dim', head_dim)
        if hasattr(attn, 'attention_head_size'):
            self._backup_param('attention_head_size', attn.attention_head_size)
            attn.attention_head_size = head_dim * 2
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        if 'attention_head_size' in self._backups:
            attn.attention_head_size = self._backups['attention_head_size']
        self._active = False


class FreezeQKVFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        q, k, v = get_qkv_projections(attn)
        self._frozen = []
        for proj in (q, k, v):
            for p in proj.parameters():
                self._backup_param(f'grad_{id(p)}', p.requires_grad)
                p.requires_grad = False
                self._frozen.append(p)
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        for p in self._frozen:
            p.requires_grad = self._backups.get(f'grad_{id(p)}', True)
        self._frozen = []
        self._active = False


QKV_FAULTS = {
    'zero_query': ZeroQueryFault,
    'zero_key': ZeroKeyFault,
    'zero_value': ZeroValueFault,
    'swapped_qk': SwappedQKFault,
    'tie_heads': TieHeadsFault,
    'wrong_head_dim': WrongHeadDimFault,
    'freeze_qkv': FreezeQKVFault,
}


def create_qkv_fault(name: str, **kwargs) -> AttentionFault:
    return QKV_FAULTS[name](**kwargs)
