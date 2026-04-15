import torch
import torch.nn as nn
from .base_fault import AttentionFault
from .attention_utils import get_qkv_projections


class WrongVariantFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        heads_attr = None
        for attr in ('num_attention_heads', 'n_heads'):
            if hasattr(attn, attr):
                heads_attr = attr
                break
        if heads_attr:
            self._backup_param('num_heads', getattr(attn, heads_attr))
            setattr(attn, heads_attr, 1)
            if hasattr(attn, 'attention_head_size'):
                self._backup_param('head_size', attn.attention_head_size)
                attn.attention_head_size = attn.all_head_size
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        for attr in ('num_attention_heads', 'n_heads'):
            if hasattr(attn, attr) and 'num_heads' in self._backups:
                setattr(attn, attr, self._backups['num_heads'])
                break
        if 'head_size' in self._backups:
            attn.attention_head_size = self._backups['head_size']
        self._active = False


class CausalInBidirectionalFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        self._orig_forward = attn.forward
        self._active = True

        def patched_forward(*args, **kwargs):
            result = self._orig_forward(*args, **kwargs)
            hidden = args[0] if args else kwargs.get('hidden_states')
            if hidden is not None:
                seq_len = hidden.shape[1]
                device = hidden.device
                causal = torch.triu(
                    torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
                )
                if isinstance(result, tuple) and result[0] is not None:
                    out = result[0] + causal.unsqueeze(0).unsqueeze(0)[:, :, :result[0].shape[-2], :result[0].shape[-1]] * 0
                    return result
            return result

        self._backup_param('forward', self._orig_forward)
        attn.forward = patched_forward
        self._attn_ref = attn

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._attn_ref.forward = self._backups['forward']
        self._active = False


VARIANT_FAULTS = {
    'wrong_variant': WrongVariantFault,
    'causal_in_bidirectional': CausalInBidirectionalFault,
}


def create_variant_fault(name: str, **kwargs) -> AttentionFault:
    return VARIANT_FAULTS[name](**kwargs)
