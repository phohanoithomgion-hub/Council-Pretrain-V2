import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch._dynamo

torch._dynamo.config.recompile_limit = 16
torch._dynamo.config.cache_size_limit = 32
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2Native
    _HAS_MAMBA2 = True
except ImportError:
    _Mamba2Native = None
    _HAS_MAMBA2 = False

try:
    from mamba_ssm.utils.generation import InferenceParams as _OfficialInferenceParams
except ImportError:
    _OfficialInferenceParams = None


def _str_to_dtype(s):
    if s is None:
        return None
    if isinstance(s, torch.dtype):
        return s
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(str(s).lower(), None)


def _create_inference_params(max_seqlen, max_batch_size, num_heads, head_dim, n_layer):
    if _OfficialInferenceParams is not None:
        try:
            return _OfficialInferenceParams(
                max_seqlen=max_seqlen,
                max_batch_size=max_batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                n_layer=n_layer,
            )
        except TypeError:
            try:
                return _OfficialInferenceParams(
                    max_seqlen=max_seqlen,
                    max_batch_size=max_batch_size,
                )
            except TypeError:
                pass

    class _Fallback:
        pass

    fp = _Fallback()
    fp.max_seqlen = max_seqlen
    fp.max_batch_size = max_batch_size
    fp.seqlen_offset = 0
    fp.key_value_memory_dict = {}
    return fp


class CausalConv1d(nn.Module):
    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _build_cache_from_real(self, x, attention_mask):
        B, T, D = x.shape
        k = self.kernel_size
        new_cache = torch.zeros(B, k - 1, D, device=x.device, dtype=x.dtype)
        if k <= 1 or T == 0:
            return new_cache
        am_bool = attention_mask[:, :T].bool()
        lengths = am_bool.long().sum(dim=1)
        rank = am_bool.long().cumsum(dim=1)
        valid_src = am_bool & (rank > (lengths.unsqueeze(1) - (k - 1)))
        if valid_src.any():
            b_idx, t_idx = valid_src.nonzero(as_tuple=True)
            dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 2)
            new_cache[b_idx, dst_idx] = x[b_idx, t_idx].detach()
        return new_cache

    def _build_seq_cache_from_real(self, seq_idx, attention_mask, T):
        B = seq_idx.shape[0]
        k = self.kernel_size
        new_seq_cache = torch.full((B, k - 1), -1, device=seq_idx.device, dtype=torch.int32)
        if k <= 1 or T == 0:
            return new_seq_cache
        seq_t = seq_idx[:, :T].to(torch.int32)
        if attention_mask is not None:
            am_bool = attention_mask[:, :T].bool()
            lengths = am_bool.long().sum(dim=1)
            rank = am_bool.long().cumsum(dim=1)
            valid_src = am_bool & (rank > (lengths.unsqueeze(1) - (k - 1)))
            if valid_src.any():
                b_idx, t_idx = valid_src.nonzero(as_tuple=True)
                dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 2)
                new_seq_cache[b_idx, dst_idx] = seq_t[b_idx, t_idx]
        else:
            if T >= k - 1:
                new_seq_cache = seq_t[:, -(k - 1):].detach()
            elif T > 0:
                new_seq_cache[:, (k - 1 - T):] = seq_t.detach()
        return new_seq_cache

    def _run_conv_with_seq(self, x_seq, seq_arr, T):
        B, _, D = x_seq.shape
        k = self.kernel_size
        xc = F.pad(x_seq.transpose(1, 2), (k - 1, 0))
        seq_padded = F.pad(seq_arr.long(), (k - 1, 0), value=-1)
        q_seq = seq_padded[:, k - 1:]
        weight = self.conv.weight
        out = torch.zeros(B, D, T, device=x_seq.device, dtype=x_seq.dtype)
        for j in range(k):
            tap = xc[:, :, j:j + T]
            k_seq = seq_padded[:, j:j + T]
            m = (q_seq == k_seq) & (k_seq >= 0)
            out = out + weight[:, 0, j].view(1, D, 1) * tap * m.unsqueeze(1).to(tap.dtype)
        return out.transpose(1, 2)

    def forward(self, x, conv_cache=None, seq_idx=None, attention_mask=None,
                seq_cache=None, return_seq_cache=False):
        B, T, D = x.shape
        k = self.kernel_size
        new_seq_cache = None
        if conv_cache is None:
            if attention_mask is not None and T > 0:
                am_bool = attention_mask[:, :T].bool()
                am_f = am_bool.unsqueeze(-1).to(x.dtype)
                x = x * am_f
                residual = x
                real_pos = am_bool.long().cumsum(dim=1) - 1
                pos_safe = torch.where(am_bool, real_pos, torch.full_like(real_pos, T))
                x_comp = torch.zeros(B, T + 1, D, device=x.device, dtype=x.dtype).scatter(
                    1, pos_safe.unsqueeze(-1).expand(-1, -1, D), x)[:, :T, :]
                if seq_idx is not None:
                    seq_comp = torch.full((B, T + 1), -1, device=seq_idx.device, dtype=torch.int32).scatter(
                        1, pos_safe, seq_idx[:, :T].to(torch.int32))[:, :T]
                    xc_comp = self._run_conv_with_seq(x_comp, seq_comp, T)
                else:
                    xc_pad = F.pad(x_comp.transpose(1, 2), (k - 1, 0))
                    xc_comp = self.conv(xc_pad)[:, :, :T].transpose(1, 2)
                xc = F.pad(xc_comp, (0, 0, 0, 1)).gather(1, pos_safe.unsqueeze(-1).expand(-1, -1, D))
                xc = xc * am_f
                if k > 1:
                    lengths = am_bool.long().sum(dim=1)
                    cache_offs = lengths.unsqueeze(1) - (k - 1) + torch.arange(k - 1, device=x.device).unsqueeze(0)
                    valid_c = cache_offs >= 0
                    new_cache = x_comp.gather(
                        1, cache_offs.clamp(min=0, max=max(T - 1, 0)).unsqueeze(-1).expand(-1, -1, D)
                    ).detach() * valid_c.unsqueeze(-1).to(x.dtype)
                else:
                    new_cache = torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)
            else:
                residual = x
                if seq_idx is not None:
                    xc = self._run_conv_with_seq(x, seq_idx[:, :T], T)
                else:
                    xc_pad = F.pad(x.transpose(1, 2), (k - 1, 0))
                    xc = self.conv(xc_pad)[:, :, :T].transpose(1, 2)
                if T >= k - 1:
                    new_cache = x[:, -(k - 1):].detach()
                else:
                    pad = torch.zeros(B, k - 1 - T, D, device=x.device, dtype=x.dtype)
                    new_cache = torch.cat([pad, x.detach()], dim=1)
            if return_seq_cache and seq_idx is not None:
                new_seq_cache = self._build_seq_cache_from_real(seq_idx, attention_mask, T)
        else:
            residual = x
            cat_x = torch.cat([conv_cache, x], dim=1)
            if seq_cache is not None and seq_idx is not None:
                q_seq = seq_idx[:, :T].to(torch.int32)
                tap_seq = torch.cat([seq_cache.to(torch.int32), q_seq], dim=1)
                if T == 1:
                    valid_taps = (tap_seq == q_seq) & (tap_seq >= 0)
                    xc_in = (cat_x * valid_taps.unsqueeze(-1).to(x.dtype)).transpose(1, 2)
                    xc = self.conv(xc_in)[:, :, -1:].transpose(1, 2)
                else:
                    xc_t = cat_x.transpose(1, 2)
                    weight = self.conv.weight
                    out = torch.zeros(B, D, T, device=x.device, dtype=x.dtype)
                    for j in range(k):
                        tap = xc_t[:, :, j:j + T]
                        k_seq = tap_seq[:, j:j + T]
                        m = (q_seq == k_seq) & (k_seq >= 0)
                        out = out + weight[:, 0, j].view(1, D, 1) * tap * m.unsqueeze(1).to(tap.dtype)
                    xc = out.transpose(1, 2)
                if return_seq_cache:
                    new_seq_cache = tap_seq[:, -(k - 1):].detach() if k > 1 else tap_seq[:, :0]
            else:
                xc = self.conv(cat_x.transpose(1, 2))[:, :, -T:].transpose(1, 2)
            new_cache = cat_x[:, -(k - 1):].detach() if k > 1 else cat_x[:, :0].detach()
        xc = self.proj(xc) * F.silu(self.gate(xc))
        out = self.norm(residual + xc)
        if return_seq_cache:
            return out, new_cache, new_seq_cache
        return out, new_cache


def _causal_mask(T_q, T_kv, past_len, device, window_size=None):
    q_pos = torch.arange(past_len, past_len + T_q, device=device)
    k_pos = torch.arange(T_kv, device=device)
    mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    if window_size is not None:
        win = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) <= window_size
        mask = mask & win
    return mask


def _seq_boundary_mask(seq_idx, T_q, T_kv, past_len, device, past_seq_idx=None):
    if seq_idx is None:
        return None
    if past_len == 0:
        if seq_idx.size(1) < T_q or seq_idx.size(1) < T_kv:
            return None
        q_seq = seq_idx[:, :T_q]
        k_seq = seq_idx[:, :T_kv]
    else:
        if past_seq_idx is None or seq_idx.size(1) < T_q:
            return None
        q_seq = seq_idx[:, :T_q]
        k_seq = torch.cat([past_seq_idx, q_seq], dim=1)
        if k_seq.size(1) != T_kv:
            return None
    same = (q_seq.unsqueeze(-1) == k_seq.unsqueeze(-2)) & (q_seq.unsqueeze(-1) >= 0) & (k_seq.unsqueeze(-2) >= 0)
    return same.unsqueeze(1)


def _safe_sdpa(q, k, v, attn_mask):
    if attn_mask.size(-1) == 0:
        return torch.zeros_like(q)
    has_valid = attn_mask.any(dim=-1, keepdim=True)
    safe_mask = attn_mask.clone()
    safe_mask[..., 0] = safe_mask[..., 0] | (~has_valid.squeeze(-1))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask, is_causal=False)
    return torch.where(has_valid, out, torch.zeros_like(out))


class GQAAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim, window_size, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def _expand_kv(self, k, v, B):
        if self.num_groups == 1:
            return k, v
        k_e = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        v_e = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        return k_e, v_e

    def _core(self, x, past_kv, seq_idx, attention_mask, return_importance,
              past_seq_idx=None, q_pos=None, past_kv_pos=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)
        T_q, T_kv = q.shape[2], k.shape[2]
        k_e, v_e = self._expand_kv(k, v, B)
        if q_pos is not None or past_kv_pos is not None:
            if q_pos is None:
                q_pos_t = torch.arange(past_len, past_len + T_q, device=x.device).unsqueeze(0).expand(B, T_q)
            else:
                q_pos_t = q_pos
            if past_kv_pos is not None:
                k_pos_t = torch.cat([past_kv_pos, q_pos_t], dim=1)
            elif past_len == 0:
                k_pos_t = q_pos_t
            else:
                past_pos = torch.arange(past_len, device=x.device).unsqueeze(0).expand(B, past_len)
                k_pos_t = torch.cat([past_pos, q_pos_t], dim=1)
            causal = (k_pos_t.unsqueeze(1) >= 0) & (q_pos_t.unsqueeze(2) >= 0) & (k_pos_t.unsqueeze(1) <= q_pos_t.unsqueeze(2))
            if self.window_size is not None:
                causal = causal & ((q_pos_t.unsqueeze(2) - k_pos_t.unsqueeze(1)) <= self.window_size)
            attn_bool = causal.unsqueeze(1)
        else:
            q_pos_t = None
            causal = _causal_mask(T_q, T_kv, past_len, x.device, self.window_size)
            attn_bool = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        sb = _seq_boundary_mask(seq_idx, T_q, T_kv, past_len, x.device, past_seq_idx=past_seq_idx)
        if sb is not None:
            attn_bool = attn_bool & sb
        if attention_mask is not None and attention_mask.size(1) == T_kv:
            attn_bool = attn_bool & attention_mask[:, None, None, :].bool()
        if return_importance:
            has_valid = attn_bool.any(dim=-1, keepdim=True)
            safe_mask = attn_bool.clone()
            safe_mask[..., 0] = safe_mask[..., 0] | (~has_valid.squeeze(-1))
            scores = torch.matmul(q, k_e.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~safe_mask, float('-inf'))
            attn_w = F.softmax(scores.float(), dim=-1)
            attn_w = torch.where(has_valid, attn_w, torch.zeros_like(attn_w))
            out = torch.matmul(attn_w.to(v_e.dtype), v_e)
            if attention_mask is not None and attention_mask.size(1) >= past_len + T_q:
                q_mask = attention_mask[:, past_len:past_len + T_q].float()
                if q_pos_t is not None:
                    q_mask = q_mask * (q_pos_t >= 0).float()
                C = (attn_w * q_mask[:, None, :, None]).sum(dim=(1, 2))
            elif q_pos_t is not None:
                C = (attn_w * (q_pos_t >= 0).float()[:, None, :, None]).sum(dim=(1, 2))
            else:
                C = attn_w.sum(dim=(1, 2))
            if attention_mask is not None:
                C = C * attention_mask.float()[:, :T_kv]
        else:
            out = _safe_sdpa(q, k_e, v_e, attn_bool)
            C = None
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.head_dim)
        return self.out_proj(out), new_kv, C

    def forward(self, x, past_kv=None, seq_idx=None, attention_mask=None,
                return_importance=False, past_seq_idx=None, q_pos=None, past_kv_pos=None):
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and past_kv is None:
            def _fn(inp):
                return self._core(inp, None, seq_idx, attention_mask, return_importance,
                                  None, q_pos, None)
            return checkpoint(_fn, x, use_reentrant=True)
        return self._core(x, past_kv, seq_idx, attention_mask, return_importance,
                          past_seq_idx=past_seq_idx, q_pos=q_pos, past_kv_pos=past_kv_pos)


class ThoughtGQAAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim,
                 block_size, top_k_heads, max_blocks, max_full, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.block_size = block_size
        self.top_k_heads = top_k_heads
        self.max_blocks = max_blocks
        self.max_full = max_full
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def _build_head_mask(self, q_mean, k_blocks_mean, selected, advanced):
        B, H, hd = q_mean.shape
        NB = k_blocks_mean.shape[1]
        kv_h = k_blocks_mean.shape[2]
        ng = H // kv_h
        k_h = k_blocks_mean.unsqueeze(3).expand(-1, -1, -1, ng, -1).reshape(B, NB, H, hd)
        scores = torch.einsum('bhd,bnhd->bhn', q_mean.float(), k_h.float())
        K = min(self.top_k_heads, H)
        topk_idx = scores.topk(K, dim=1).indices
        head_ok = torch.zeros(B, H, NB, dtype=torch.bool, device=q_mean.device)
        head_ok.scatter_(1, topk_idx, torch.ones_like(topk_idx, dtype=torch.bool))
        head_ok = torch.where(advanced.unsqueeze(1), torch.ones_like(head_ok), head_ok)
        head_ok = head_ok & selected.unsqueeze(1)
        return head_ok

    def _expand_kv(self, k, v, B):
        if self.num_groups == 1:
            return k, v
        k_e = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        v_e = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1).reshape(
            B, self.num_heads, -1, self.head_dim)
        return k_e, v_e

    def _core_prefill(self, x, seq_idx, attention_mask, selected, advanced, build_cache,
                      real_pos=None, selected_per_seq=None, advanced_per_seq=None):
        B, T, D = x.shape
        device = x.device
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if selected is None:
            NB = (T + self.block_size - 1) // self.block_size
            selected = torch.ones(B, NB, dtype=torch.bool, device=device)
            advanced = torch.zeros(B, NB, dtype=torch.bool, device=device)
        else:
            NB = selected.shape[1]

        if selected_per_seq is not None and seq_idx is not None:
            num_seqs = selected_per_seq.shape[1]
            pos_arr = real_pos if real_pos is not None else torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            pos_safe = pos_arr.clamp(min=0, max=max(NB * self.block_size - 1, 0))
            valid_tok = (pos_arr >= 0) & (pos_arr < NB * self.block_size)
            if attention_mask is not None:
                valid_tok = valid_tok & attention_mask[:, :T].bool()
            tok_blk = pos_safe // self.block_size
            seq_eff = seq_idx[:, :T].to(torch.int32)
            head_ok_tok = torch.zeros(B, self.num_heads, T, dtype=torch.bool, device=device)
            k_t = k.transpose(1, 2).float()
            q_f = q.float()
            blk_idx_exp = tok_blk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_kv_heads, self.head_dim)
            for s in range(num_seqs):
                s_val = s + 1
                mask_s = (seq_eff == s_val) & valid_tok
                v_fs = mask_s.float()
                k_sum_s = torch.zeros(B, NB, self.num_kv_heads, self.head_dim, device=device, dtype=torch.float32).scatter_add(
                    1, blk_idx_exp, k_t * v_fs.unsqueeze(-1).unsqueeze(-1))
                cnt_s = torch.zeros(B, NB, device=device, dtype=torch.float32).scatter_add(1, tok_blk, v_fs)
                valid_blk_s = cnt_s > 0
                k_bm_s = k_sum_s / cnt_s.clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)
                q_mean_s = (q_f * v_fs[:, None, :, None]).sum(dim=2) / v_fs.sum(dim=1).clamp(min=1.0)[:, None, None]
                head_ok_s = self._build_head_mask(
                    q_mean_s, k_bm_s, selected_per_seq[:, s], advanced_per_seq[:, s] & valid_blk_s)
                head_ok_s = head_ok_s & valid_blk_s.unsqueeze(1)
                tok_s = head_ok_s.gather(2, tok_blk.unsqueeze(1).expand(-1, self.num_heads, -1)) & mask_s.unsqueeze(1)
                head_ok_tok = torch.where(mask_s.unsqueeze(1), tok_s, head_ok_tok)
            if real_pos is not None:
                causal = (real_pos.unsqueeze(1) >= 0) & (real_pos.unsqueeze(2) >= 0) & (real_pos.unsqueeze(1) <= real_pos.unsqueeze(2))
                mask = head_ok_tok[:, :, None, :] & causal.unsqueeze(1)
            else:
                causal = _causal_mask(T, T, 0, device, None)
                mask = head_ok_tok[:, :, None, :] & causal.unsqueeze(0).unsqueeze(0)
        elif real_pos is not None:
            pos_safe = real_pos.clamp(min=0, max=max(NB * self.block_size - 1, 0))
            valid_tok = (real_pos >= 0) & (real_pos < NB * self.block_size)
            if attention_mask is not None:
                valid_tok = valid_tok & attention_mask[:, :T].bool()
            tok_blk = pos_safe // self.block_size
            v_f = valid_tok.float()
            k_masked = k.transpose(1, 2).float() * v_f.unsqueeze(-1).unsqueeze(-1)
            k_sum_buf = torch.zeros(B, NB, self.num_kv_heads, self.head_dim, device=device, dtype=torch.float32).scatter_add(
                1, tok_blk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_kv_heads, self.head_dim), k_masked)
            cnt_buf = torch.zeros(B, NB, device=device, dtype=torch.float32).scatter_add(1, tok_blk, v_f)
            valid_blk = cnt_buf > 0
            k_bm = k_sum_buf / cnt_buf.clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)
            q_mean = (q.float() * v_f[:, None, :, None]).sum(dim=2) / v_f.sum(dim=1).clamp(min=1.0)[:, None, None]
            head_ok = self._build_head_mask(q_mean, k_bm, selected, advanced & valid_blk)
            head_ok = head_ok & valid_blk.unsqueeze(1)
            head_ok_tok = head_ok.gather(2, tok_blk.unsqueeze(1).expand(-1, self.num_heads, -1))
            head_ok_tok = head_ok_tok & valid_tok.unsqueeze(1)
            causal = (real_pos.unsqueeze(1) >= 0) & (real_pos.unsqueeze(2) >= 0) & (real_pos.unsqueeze(1) <= real_pos.unsqueeze(2))
            mask = head_ok_tok[:, :, None, :] & causal.unsqueeze(1)
        else:
            pad_len = NB * self.block_size - T
            k_p = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
            k_blocks = k_p.view(B, self.num_kv_heads, NB, self.block_size, self.head_dim).permute(0, 2, 1, 3, 4)
            if attention_mask is not None:
                am = attention_mask[:, :T].float()
            else:
                am = torch.ones(B, T, device=device, dtype=torch.float32)
            if pad_len > 0:
                am = F.pad(am, (0, pad_len), value=0.0)
            am_blk = am.view(B, NB, self.block_size)
            raw_counts = am_blk.sum(-1)
            valid_blk = raw_counts > 0
            safe_counts = raw_counts.clamp(min=1.0)
            k_bm = (k_blocks.float() * am_blk.unsqueeze(2).unsqueeze(-1)).sum(3) / safe_counts.unsqueeze(-1).unsqueeze(-1)
            if attention_mask is not None:
                q_am = am[:, :T]
                q_mean = (q.float() * q_am[:, None, :, None]).sum(dim=2) / q_am.sum(dim=1).clamp(min=1.0)[:, None, None]
            else:
                q_mean = q.float().mean(dim=2)
            head_ok = self._build_head_mask(q_mean, k_bm, selected, advanced & valid_blk)
            head_ok = head_ok & valid_blk.unsqueeze(1)
            head_ok_tok = head_ok.repeat_interleave(self.block_size, dim=-1)[:, :, :T]
            causal = _causal_mask(T, T, 0, device, None)
            mask = head_ok_tok[:, :, None, :] & causal.unsqueeze(0).unsqueeze(0)

        sb = _seq_boundary_mask(seq_idx, T, T, 0, device)
        if sb is not None:
            mask = mask & sb
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :T].bool()

        k_e, v_e = self._expand_kv(k, v, B)
        out = _safe_sdpa(q, k_e, v_e, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        new_cache = None
        if build_cache:
            new_cache = self._build_cache_from_full(k, v, T, selected, advanced,
                                                    attention_mask, device, seq_idx=seq_idx,
                                                    real_pos=real_pos)
        return out, new_cache

    def forward_prefill(self, x, seq_idx=None, attention_mask=None,
                        selected=None, advanced=None, build_cache=False,
                        real_pos=None, selected_per_seq=None, advanced_per_seq=None):
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and not build_cache:
            def _fn(inp):
                return self._core_prefill(
                    inp, seq_idx, attention_mask, selected, advanced, False,
                    real_pos=real_pos, selected_per_seq=selected_per_seq,
                    advanced_per_seq=advanced_per_seq)[0]
            return checkpoint(_fn, x, use_reentrant=True), None
        return self._core_prefill(
            x, seq_idx, attention_mask, selected, advanced, build_cache,
            real_pos=real_pos, selected_per_seq=selected_per_seq,
            advanced_per_seq=advanced_per_seq)

    def _build_cache_from_full(self, k, v, T, selected, advanced, attention_mask, device,
                               seq_idx=None, real_pos=None):
        B, kv_heads, _, hd = k.shape
        NB = max(selected.shape[1], self.max_blocks)
        selected_p = torch.zeros(B, NB, dtype=torch.bool, device=device)
        advanced_p = torch.zeros(B, NB, dtype=torch.bool, device=device)
        S = selected.shape[1]
        selected_p[:, :S] = selected
        advanced_p[:, :S] = advanced

        total_cap = NB * self.block_size
        if total_cap < T:
            raise ValueError(f"T={T} exceeds NB*block_size={total_cap}")

        if real_pos is not None:
            valid_tok = (real_pos >= 0) & (real_pos < total_cap)
            if attention_mask is not None:
                valid_tok = valid_tok & attention_mask[:, :T].bool()
            k_bank_flat = torch.zeros(B, total_cap, kv_heads, hd, device=device, dtype=k.dtype)
            v_bank_flat = torch.zeros(B, total_cap, kv_heads, hd, device=device, dtype=v.dtype)
            blk_mask_flat = torch.zeros(B, total_cap, device=device, dtype=torch.bool)
            if valid_tok.any():
                b_idx, t_idx = valid_tok.nonzero(as_tuple=True)
                pos_idx = real_pos[b_idx, t_idx]
                k_bank_flat[b_idx, pos_idx] = k.transpose(1, 2)[b_idx, t_idx]
                v_bank_flat[b_idx, pos_idx] = v.transpose(1, 2)[b_idx, t_idx]
                blk_mask_flat[b_idx, pos_idx] = True
            k_bank = k_bank_flat.view(B, NB, self.block_size, kv_heads, hd).contiguous()
            v_bank = v_bank_flat.view(B, NB, self.block_size, kv_heads, hd).contiguous()
            am_blk_bool = blk_mask_flat.view(B, NB, self.block_size).contiguous()
            row_seq_lens = valid_tok.long().sum(dim=1)
        else:
            pad_len = total_cap - T
            k_p = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
            v_p = F.pad(v, (0, 0, 0, pad_len)) if pad_len > 0 else v
            k_bank = k_p.view(B, kv_heads, NB, self.block_size, hd).permute(0, 2, 3, 1, 4).contiguous()
            v_bank = v_p.view(B, kv_heads, NB, self.block_size, hd).permute(0, 2, 3, 1, 4).contiguous()
            if attention_mask is not None:
                am = attention_mask[:, :T].float()
            else:
                am = torch.ones(B, T, device=device, dtype=torch.float32)
            if pad_len > 0:
                am = F.pad(am, (0, pad_len), value=0.0)
            am_blk_bool = am.view(B, NB, self.block_size).bool()
            m5 = am_blk_bool.unsqueeze(-1).unsqueeze(-1)
            k_bank = torch.where(m5, k_bank, torch.zeros_like(k_bank))
            v_bank = torch.where(m5, v_bank, torch.zeros_like(v_bank))
            row_seq_lens = am_blk_bool.view(B, -1).long().sum(dim=1)

        actual_counts = am_blk_bool.long().sum(-1)

        blk_seq_ids = None
        full_seq_ids = None
        active_seq_ids = None
        if seq_idx is not None:
            seq_t = seq_idx[:, :T].to(torch.int32)
            if attention_mask is not None:
                seq_t = torch.where(attention_mask[:, :T].bool(), seq_t, torch.full_like(seq_t, -1))
            if real_pos is not None:
                blk_seq_flat = torch.full((B, total_cap), -1, device=device, dtype=torch.int32)
                valid_s = (real_pos >= 0) & (real_pos < total_cap) & (seq_t >= 0)
                if valid_s.any():
                    b_idx, t_idx = valid_s.nonzero(as_tuple=True)
                    blk_seq_flat[b_idx, real_pos[b_idx, t_idx]] = seq_t[b_idx, t_idx]
                blk_seq_ids = blk_seq_flat.view(B, NB, self.block_size)
            else:
                pad_len = total_cap - T
                seq_p = F.pad(seq_t, (0, pad_len), value=-1) if pad_len > 0 else seq_t
                blk_seq_ids = seq_p.view(B, NB, self.block_size)
            pos_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            valid_seq = seq_t >= 0
            last_valid_pos = torch.where(valid_seq, pos_grid, torch.full_like(pos_grid, -1)).max(dim=1).values.clamp(min=0)
            active_seq_ids = seq_t[torch.arange(B, device=device), last_valid_pos]
            active_seq_ids = torch.where(valid_seq.any(dim=1), active_seq_ids, torch.full_like(active_seq_ids, -1))
            sum_mask_b = am_blk_bool & (blk_seq_ids == active_seq_ids.view(B, 1, 1)) & (active_seq_ids.view(B, 1, 1) >= 0)
            full_seq_ids = torch.full((B, self.max_full, self.block_size), -1, device=device, dtype=torch.int32)
        else:
            sum_mask_b = am_blk_bool

        sum_counts = sum_mask_b.long().sum(-1)
        safe_sum_counts = sum_counts.clamp(min=1).float()
        sm5 = sum_mask_b.unsqueeze(-1).unsqueeze(-1).float()

        k_sum_fp32 = (k_bank.float() * sm5).sum(2) / safe_sum_counts.unsqueeze(-1).unsqueeze(-1)
        v_sum_fp32 = (v_bank.float() * sm5).sum(2) / safe_sum_counts.unsqueeze(-1).unsqueeze(-1)

        cache = {
            "k_bank": k_bank,
            "v_bank": v_bank,
            "blk_mask": am_blk_bool,
            "blk_seq_ids": blk_seq_ids,
            "active_seq_ids": active_seq_ids,
            "k_sum": k_sum_fp32.to(k.dtype),
            "v_sum": v_sum_fp32.to(v.dtype),
            "k_sum_fp32": k_sum_fp32,
            "v_sum_fp32": v_sum_fp32,
            "blk_lens": actual_counts,
            "sum_lens": sum_counts,
            "k_full": torch.zeros(B, self.max_full, self.block_size, kv_heads, hd,
                                  device=device, dtype=k.dtype),
            "v_full": torch.zeros(B, self.max_full, self.block_size, kv_heads, hd,
                                  device=device, dtype=v.dtype),
            "full_mask": torch.zeros(B, self.max_full, self.block_size, device=device, dtype=torch.bool),
            "full_seq_ids": full_seq_ids,
            "full_ids": torch.full((B, self.max_full), -1, device=device, dtype=torch.long),
            "full_lens": torch.zeros(B, self.max_full, device=device, dtype=torch.long),
            "seq_len": T,
            "row_seq_lens": row_seq_lens,
            "selected": selected_p,
            "advanced": advanced_p,
        }

        for b in range(B):
            sel_blocks = selected_p[b].nonzero(as_tuple=True)[0][:self.max_full].tolist()
            for slot, blk in enumerate(sel_blocks):
                n = int(actual_counts[b, blk].item())
                cache["full_ids"][b, slot] = blk
                cache["full_lens"][b, slot] = n
                cache["full_mask"][b, slot] = am_blk_bool[b, blk]
                cache["k_full"][b, slot] = k_bank[b, blk]
                cache["v_full"][b, slot] = v_bank[b, blk]
                if full_seq_ids is not None:
                    full_seq_ids[b, slot] = blk_seq_ids[b, blk]
        return cache

    def _expand_cache_blocks(self, cache, new_nb):
        old_nb = cache["selected"].shape[1]
        if new_nb <= old_nb:
            return
        extra = new_nb - old_nb
        B = cache["selected"].shape[0]
        bsz = self.block_size
        kv_h = self.num_kv_heads
        hd = self.head_dim
        device = cache["selected"].device
        cache["k_bank"] = torch.cat(
            [cache["k_bank"], torch.zeros(B, extra, bsz, kv_h, hd, device=device, dtype=cache["k_bank"].dtype)],
            dim=1,
        )
        cache["v_bank"] = torch.cat(
            [cache["v_bank"], torch.zeros(B, extra, bsz, kv_h, hd, device=device, dtype=cache["v_bank"].dtype)],
            dim=1,
        )
        cache["blk_mask"] = torch.cat(
            [cache["blk_mask"], torch.zeros(B, extra, bsz, device=device, dtype=torch.bool)],
            dim=1,
        )
        if cache["blk_seq_ids"] is not None:
            cache["blk_seq_ids"] = torch.cat(
                [cache["blk_seq_ids"], torch.full((B, extra, bsz), -1, device=device, dtype=torch.int32)],
                dim=1,
            )
        cache["k_sum"] = torch.cat(
            [cache["k_sum"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=cache["k_sum"].dtype)],
            dim=1,
        )
        cache["v_sum"] = torch.cat(
            [cache["v_sum"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=cache["v_sum"].dtype)],
            dim=1,
        )
        cache["k_sum_fp32"] = torch.cat(
            [cache["k_sum_fp32"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=torch.float32)],
            dim=1,
        )
        cache["v_sum_fp32"] = torch.cat(
            [cache["v_sum_fp32"], torch.zeros(B, extra, kv_h, hd, device=device, dtype=torch.float32)],
            dim=1,
        )
        cache["blk_lens"] = torch.cat(
            [cache["blk_lens"], torch.zeros(B, extra, device=device, dtype=torch.long)],
            dim=1,
        )
        cache["sum_lens"] = torch.cat(
            [cache["sum_lens"], torch.zeros(B, extra, device=device, dtype=torch.long)],
            dim=1,
        )
        cache["selected"] = torch.cat(
            [cache["selected"], torch.zeros(B, extra, device=device, dtype=torch.bool)],
            dim=1,
        )
        cache["advanced"] = torch.cat(
            [cache["advanced"], torch.zeros(B, extra, device=device, dtype=torch.bool)],
            dim=1,
        )

    def _reeval_slots(self, cache, reeval_mask=None):
        B = cache["full_ids"].shape[0]
        F = cache["full_ids"].shape[1]
        NB = cache["selected"].shape[1]
        device = cache["full_ids"].device
        sel = cache["selected"]

        old_ids = cache["full_ids"]
        old_ids_safe = old_ids.clamp(min=0)
        blk_is_sel = sel.gather(1, old_ids_safe)
        keep = blk_is_sel & (old_ids >= 0)

        new_ids = torch.where(keep, old_ids, torch.full_like(old_ids, -1))
        free = (new_ids == -1)

        has_full = (new_ids.unsqueeze(-1) == torch.arange(NB, device=device).view(1, 1, NB)).any(1)
        need_alloc = sel & ~has_full
        need_rank = torch.cumsum(need_alloc.long(), dim=-1) - 1
        free_rank = torch.cumsum(free.long(), dim=-1) - 1

        need_rank_s = need_rank.clone().float()
        need_rank_s[~need_alloc] = float('inf')
        sorted_blocks = need_rank_s.argsort(dim=-1)
        free_rank_s = free_rank.clone().float()
        free_rank_s[~free] = float('inf')
        sorted_slots = free_rank_s.argsort(dim=-1)

        num_need = need_alloc.sum(-1)
        num_free = free.sum(-1)
        num_assign = torch.minimum(num_need, num_free)

        slot_assignment = new_ids.clone()
        b_idx = torch.arange(B, device=device)
        for r in range(F):
            valid = r < num_assign
            if not valid.any():
                break
            v_b = b_idx[valid]
            v_s = sorted_slots[:, r][valid]
            v_blk = sorted_blocks[:, r][valid]
            slot_assignment[v_b, v_s] = v_blk

        if reeval_mask is not None:
            slot_assignment = torch.where(reeval_mask.unsqueeze(1), slot_assignment, old_ids)

        if reeval_mask is not None:
            newly_assigned = (new_ids == -1) & (slot_assignment >= 0) & reeval_mask.unsqueeze(1)
            still_free = (slot_assignment == -1) & reeval_mask.unsqueeze(1)
        else:
            newly_assigned = (new_ids == -1) & (slot_assignment >= 0)
            still_free = slot_assignment == -1

        if newly_assigned.any():
            nb_idx, ns_idx = newly_assigned.nonzero(as_tuple=True)
            blk_idx = slot_assignment[nb_idx, ns_idx]
            cache["k_full"][nb_idx, ns_idx] = cache["k_bank"][nb_idx, blk_idx]
            cache["v_full"][nb_idx, ns_idx] = cache["v_bank"][nb_idx, blk_idx]
            cache["full_mask"][nb_idx, ns_idx] = cache["blk_mask"][nb_idx, blk_idx]
            if cache["full_seq_ids"] is not None and cache["blk_seq_ids"] is not None:
                cache["full_seq_ids"][nb_idx, ns_idx] = cache["blk_seq_ids"][nb_idx, blk_idx]

        if still_free.any():
            fb_idx, fs_idx = still_free.nonzero(as_tuple=True)
            cache["full_mask"][fb_idx, fs_idx] = False
            if cache["full_seq_ids"] is not None:
                cache["full_seq_ids"][fb_idx, fs_idx] = -1

        cache["full_ids"] = slot_assignment
        cache["full_lens"] = cache["full_mask"].long().sum(-1)

    def _write_token_to_cache(self, cache, blk, offset, k_t, v_t, cur_seq_ids=None):
        B = k_t.shape[0]
        device = k_t.device
        b_all = torch.arange(B, device=device)
        if isinstance(blk, int):
            blk_b = torch.full((B,), blk, device=device, dtype=torch.long)
        elif blk.dim() == 0:
            blk_b = blk.to(device=device, dtype=torch.long).expand(B)
        else:
            blk_b = blk.to(device=device, dtype=torch.long)
        if isinstance(offset, int):
            off_b = torch.full((B,), offset, device=device, dtype=torch.long)
        elif offset.dim() == 0:
            off_b = offset.to(device=device, dtype=torch.long).expand(B)
        else:
            off_b = offset.to(device=device, dtype=torch.long)

        was_valid = cache["blk_mask"][b_all, blk_b, off_b]
        if cur_seq_ids is not None and cache["blk_seq_ids"] is not None:
            was_valid_sum = was_valid & (cache["blk_seq_ids"][b_all, blk_b, off_b] == cur_seq_ids)
            cache["blk_seq_ids"][b_all, blk_b, off_b] = cur_seq_ids
        else:
            was_valid_sum = was_valid
        cache["k_bank"][b_all, blk_b, off_b] = k_t
        cache["v_bank"][b_all, blk_b, off_b] = v_t
        cache["blk_mask"][b_all, blk_b, off_b] = True

        add_blk = (~was_valid).long()
        cache["blk_lens"][b_all, blk_b] = cache["blk_lens"][b_all, blk_b] + add_blk

        old_sum_n = cache["sum_lens"][b_all, blk_b]
        need_add_sum = ~was_valid_sum
        new_sum_n = old_sum_n + need_add_sum.long()
        safe_new_n = new_sum_n.clamp(min=1).float().view(B, 1, 1)
        old_n_f = old_sum_n.float().view(B, 1, 1)
        new_k_sum_fp32 = torch.where(
            need_add_sum.view(B, 1, 1),
            (cache["k_sum_fp32"][b_all, blk_b] * old_n_f + k_t.float()) / safe_new_n,
            cache["k_sum_fp32"][b_all, blk_b],
        )
        new_v_sum_fp32 = torch.where(
            need_add_sum.view(B, 1, 1),
            (cache["v_sum_fp32"][b_all, blk_b] * old_n_f + v_t.float()) / safe_new_n,
            cache["v_sum_fp32"][b_all, blk_b],
        )
        cache["k_sum_fp32"][b_all, blk_b] = new_k_sum_fp32
        cache["v_sum_fp32"][b_all, blk_b] = new_v_sum_fp32
        cache["sum_lens"][b_all, blk_b] = new_sum_n
        cache["k_sum"][b_all, blk_b] = new_k_sum_fp32.to(cache["k_sum"].dtype)
        cache["v_sum"][b_all, blk_b] = new_v_sum_fp32.to(cache["v_sum"].dtype)

        full_ids = cache["full_ids"]
        match = (full_ids == blk_b.unsqueeze(1))
        has_match = match.any(-1)
        match_slot = match.int().argmax(-1)
        is_free = (full_ids == -1)
        has_free = is_free.any(-1)
        free_slot = is_free.int().argmax(-1)
        ids_for_lru = torch.where(full_ids >= 0, full_ids, torch.full_like(full_ids, 10**9))
        lru_slot = ids_for_lru.argmin(-1)
        slot = torch.where(has_match, match_slot, torch.where(has_free, free_slot, lru_slot))

        needs_init = ~has_match
        if needs_init.any():
            b_sel = b_all[needs_init]
            s_sel = slot[needs_init]
            blk_sel = blk_b[needs_init]
            cache["k_full"][b_sel, s_sel] = cache["k_bank"][b_sel, blk_sel]
            cache["v_full"][b_sel, s_sel] = cache["v_bank"][b_sel, blk_sel]
            cache["full_mask"][b_sel, s_sel] = cache["blk_mask"][b_sel, blk_sel]
            if cache["full_seq_ids"] is not None and cache["blk_seq_ids"] is not None:
                cache["full_seq_ids"][b_sel, s_sel] = cache["blk_seq_ids"][b_sel, blk_sel]
            cache["full_ids"][b_sel, s_sel] = blk_sel

        cache["k_full"][b_all, slot, off_b] = k_t
        cache["v_full"][b_all, slot, off_b] = v_t
        cache["full_mask"][b_all, slot, off_b] = True
        if cache["full_seq_ids"] is not None and cur_seq_ids is not None:
            cache["full_seq_ids"][b_all, slot, off_b] = cur_seq_ids
        cache["full_lens"][b_all, slot] = cache["full_mask"][b_all, slot].long().sum(-1)

    def _gather_decode_kv(self, cache, cur_seq_ids=None):
        B = cache["full_ids"].shape[0]
        device = cache["k_full"].device
        F = cache["full_ids"].shape[1]
        NB = cache["selected"].shape[1]
        bsz = self.block_size
        kv_h = self.num_kv_heads
        hd = self.head_dim

        offsets = torch.arange(bsz, device=device)
        full_pos = cache["full_ids"].unsqueeze(-1) * bsz + offsets
        full_valid = (cache["full_ids"].unsqueeze(-1) >= 0) & cache["full_mask"]
        if cache["full_seq_ids"] is not None and cur_seq_ids is not None:
            full_valid = full_valid & (cache["full_seq_ids"] == cur_seq_ids.view(B, 1, 1)) & (cur_seq_ids.view(B, 1, 1) >= 0)

        blocks_range = torch.arange(NB, device=device)
        has_full = (cache["full_ids"].unsqueeze(-1) == blocks_range.view(1, 1, NB)).any(1)
        need_sum = cache["selected"][:, :NB] & ~has_full & (cache["sum_lens"][:, :NB] > 0)
        sum_pos = blocks_range.unsqueeze(0) * bsz + (bsz - 1)
        sum_pos = sum_pos.expand(B, -1)

        k_full_flat = cache["k_full"].reshape(B, F * bsz, kv_h, hd)
        v_full_flat = cache["v_full"].reshape(B, F * bsz, kv_h, hd)

        k_all = torch.cat([k_full_flat, cache["k_sum"]], dim=1)
        v_all = torch.cat([v_full_flat, cache["v_sum"]], dim=1)
        pos_all = torch.cat([full_pos.reshape(B, -1), sum_pos], dim=1)
        valid_all = torch.cat([full_valid.reshape(B, -1), need_sum], dim=1)
        return k_all, v_all, pos_all, valid_all

    def forward_decode(self, x, cache, selected=None, advanced=None, reeval=False,
                       cur_seq_ids=None, reeval_mask=None):
        B, T_new, D = x.shape
        device = x.device
        past_len = cache["seq_len"]
        row_seq_lens_before = cache.get(
            "row_seq_lens",
            torch.full((B,), past_len, device=device, dtype=torch.long),
        )
        end_blk_max = int(((row_seq_lens_before + T_new - 1) // self.block_size).max().item())
        needed_nb = end_blk_max + 1
        if selected is not None:
            needed_nb = max(needed_nb, selected.shape[1])
        self._expand_cache_blocks(cache, needed_nb)
        NB = cache["selected"].shape[1]

        if cur_seq_ids is None:
            cur_seq_ids = cache.get("active_seq_ids", None)
        else:
            cache["active_seq_ids"] = cur_seq_ids

        b_all = torch.arange(B, device=device)
        if reeval_mask is None:
            reeval_rows = torch.full((B,), bool(reeval), device=device, dtype=torch.bool)
        else:
            reeval_rows = reeval_mask.bool()
        any_reeval = bool(reeval_rows.any().item())

        if any_reeval and selected is not None:
            S = selected.shape[1]
            r_col = reeval_rows.unsqueeze(1)
            cache["selected"][:, :S] = torch.where(r_col, selected, cache["selected"][:, :S])
            cache["advanced"][:, :S] = torch.where(r_col, advanced, cache["advanced"][:, :S])
            if S < NB:
                cache["selected"][reeval_rows, S:] = False
                cache["advanced"][reeval_rows, S:] = False
            for t in range(T_new):
                cache["selected"][b_all, (row_seq_lens_before + t) // self.block_size] = True
            self._reeval_slots(cache, reeval_mask=reeval_rows)
        else:
            for t in range(T_new):
                cache["selected"][b_all, (row_seq_lens_before + t) // self.block_size] = True

        q = self.q_proj(x).view(B, T_new, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        for t in range(T_new):
            pos_b = row_seq_lens_before + t
            blk_b = pos_b // self.block_size
            off_b = pos_b % self.block_size
            self._write_token_to_cache(cache, blk_b, off_b, k_new[:, :, t, :], v_new[:, :, t, :], cur_seq_ids=cur_seq_ids)

        cache["row_seq_lens"] = row_seq_lens_before + T_new
        cache["seq_len"] = past_len + T_new

        k_all, v_all, pos_all, valid_all = self._gather_decode_kv(cache, cur_seq_ids=cur_seq_ids)
        k_all = k_all.permute(0, 2, 1, 3).contiguous()
        v_all = v_all.permute(0, 2, 1, 3).contiguous()
        k_e, v_e = self._expand_kv(k_all, v_all, B)

        q_mean = q.mean(dim=2)
        head_ok = self._build_head_mask(q_mean, cache["k_sum"], cache["selected"], cache["advanced"])

        pos_blk = (pos_all // self.block_size).clamp(0, NB - 1)
        head_ok_pos = head_ok.gather(2, pos_blk.unsqueeze(1).expand(-1, self.num_heads, -1))

        q_pos = row_seq_lens_before.unsqueeze(1) + torch.arange(T_new, device=device).unsqueeze(0)
        causal = q_pos.view(B, 1, T_new, 1) >= pos_all.view(B, 1, 1, -1)
        mask = causal & valid_all.view(B, 1, 1, -1) & head_ok_pos.view(B, self.num_heads, 1, -1)

        out = _safe_sdpa(q, k_e, v_e, mask)
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.num_heads * self.head_dim)
        return self.out_proj(out), cache


class Mamba2Layer(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64,
                 ssm_cache_dtype=None, conv_cache_dtype=None):
        super().__init__()
        if not _HAS_MAMBA2:
            raise ImportError("mamba_ssm required: pip install mamba-ssm")
        self.ssm_cache_dtype = _str_to_dtype(ssm_cache_dtype)
        self.conv_cache_dtype = _str_to_dtype(conv_cache_dtype)
        kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv,
                      expand=expand, headdim=headdim)
        if self.ssm_cache_dtype is not None:
            kwargs["ssm_cache_dtype"] = self.ssm_cache_dtype
        if self.conv_cache_dtype is not None:
            kwargs["conv_cache_dtype"] = self.conv_cache_dtype
        try:
            self.mamba = _Mamba2Native(**kwargs)
        except TypeError:
            self.mamba = _Mamba2Native(d_model=d_model, d_state=d_state,
                                        d_conv=d_conv, expand=expand, headdim=headdim)

    def _ensure_cache(self, inference_params, batch_size, x_dtype):
        layer_idx = getattr(self.mamba, "layer_idx", None)
        if inference_params is None or layer_idx is None:
            return
        if layer_idx in inference_params.key_value_memory_dict:
            return
        conv_w = getattr(getattr(self.mamba, "conv1d", None), "weight", None)
        conv_base_dtype = conv_w.dtype if conv_w is not None else x_dtype
        conv_dtype = self.conv_cache_dtype
        if conv_dtype is None:
            conv_dtype = conv_base_dtype if x_dtype == conv_base_dtype else x_dtype
        ssm_dtype = self.ssm_cache_dtype if self.ssm_cache_dtype is not None else x_dtype
        if hasattr(self.mamba, "allocate_inference_cache") and conv_dtype == ssm_dtype:
            conv_state, ssm_state = self.mamba.allocate_inference_cache(
                batch_size, getattr(inference_params, "max_seqlen", 1024), dtype=conv_dtype
            )
            inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
            return
        if all(hasattr(self.mamba, attr) for attr in ("d_conv", "conv1d", "nheads", "headdim", "d_state", "in_proj")):
            conv_state = torch.zeros(
                batch_size, self.mamba.d_conv, self.mamba.conv1d.weight.shape[0],
                device=self.mamba.conv1d.weight.device, dtype=conv_dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size, self.mamba.nheads, self.mamba.headdim, self.mamba.d_state,
                device=self.mamba.in_proj.weight.device, dtype=ssm_dtype,
            )
            inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)

    def _expected_step_conv_dtype(self, x):
        if torch.is_autocast_enabled():
            try:
                return torch.get_autocast_dtype(x.device.type)
            except Exception:
                return x.dtype
        if hasattr(self.mamba, "in_proj"):
            return self.mamba.in_proj.weight.dtype
        return x.dtype

    def _mask_conv_state_seq_boundary(self, inference_params, seq_idx, attention_mask, T):
        layer_idx = getattr(self.mamba, "layer_idx", None)
        if inference_params is None or seq_idx is None or layer_idx is None:
            return
        if layer_idx not in inference_params.key_value_memory_dict:
            return
        conv_state, _ = inference_params.key_value_memory_dict[layer_idx]
        B = seq_idx.shape[0]
        state_len = conv_state.shape[-1]
        if state_len <= 0 or T <= 0:
            return
        seq_t = seq_idx[:, :T].to(torch.int32)
        tail_seq = torch.full((B, state_len), -1, device=seq_idx.device, dtype=torch.int32)
        if attention_mask is not None:
            am_bool = attention_mask[:, :T].bool()
            lengths = am_bool.long().sum(dim=1)
            rank = am_bool.long().cumsum(dim=1)
            valid_src = am_bool & (rank > (lengths.unsqueeze(1) - state_len))
            if valid_src.any():
                b_idx, t_idx = valid_src.nonzero(as_tuple=True)
                dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (state_len - 1)
                tail_seq[b_idx, dst_idx] = seq_t[b_idx, t_idx]
        else:
            if T >= state_len:
                tail_seq = seq_t[:, -state_len:]
            else:
                tail_seq[:, (state_len - T):] = seq_t
        active = tail_seq[:, -1:]
        keep = (tail_seq == active) & (tail_seq >= 0)
        conv_state.mul_(keep.unsqueeze(1).to(conv_state.dtype))

    def _build_flat_seq_idx(self, valid, lengths, seq_idx, device):
        B = lengths.size(0)
        flat_b = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), lengths)
        if seq_idx is None:
            return flat_b.to(torch.int32).unsqueeze(0).contiguous()
        flat_s = seq_idx[valid].long()
        if flat_s.numel() <= 1:
            return torch.zeros(1, flat_s.numel(), device=device, dtype=torch.int32)
        boundary = (flat_b[1:] != flat_b[:-1]) | (flat_s[1:] != flat_s[:-1])
        out = torch.zeros(flat_s.numel(), device=device, dtype=torch.int32)
        out[1:] = torch.cumsum(boundary.to(torch.int32), dim=0)
        return out.unsqueeze(0).contiguous()

    def _run_per_sample_prefill(self, x, valid, lengths, inference_params, seq_idx):
        B = x.shape[0]
        layer_idx = getattr(self.mamba, "layer_idx", None)
        self._ensure_cache(inference_params, B, x.dtype)
        out = torch.zeros_like(x)
        has_cache = (layer_idx is not None) and (layer_idx in inference_params.key_value_memory_dict)
        for b in range(B):
            L_b = int(lengths[b].item())
            if L_b <= 0:
                continue
            x_b = x[b:b + 1, valid[b], :]
            s_b = seq_idx[b:b + 1, valid[b]].to(torch.int32).contiguous() if seq_idx is not None else None
            sub_inf = _create_inference_params(
                max_seqlen=max(getattr(inference_params, "max_seqlen", L_b), L_b),
                max_batch_size=1,
                num_heads=getattr(self.mamba, "nheads", 1),
                head_dim=getattr(self.mamba, "headdim", 64),
                n_layer=1,
            )
            sub_inf.seqlen_offset = 0
            if has_cache:
                orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                sub_conv = torch.zeros(
                    1, orig_conv.shape[2], orig_conv.shape[1],
                    device=orig_conv.device, dtype=orig_conv.dtype,
                ).transpose(1, 2)
                sub_ssm = torch.zeros_like(orig_ssm[b:b + 1])
                sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
            y_b = self.mamba(x_b, inference_params=sub_inf, seq_idx=s_b)
            out[b:b + 1, valid[b], :] = y_b
            if has_cache and layer_idx in sub_inf.key_value_memory_dict:
                orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                orig_conv[b:b + 1].copy_(sub_conv)
                orig_ssm[b:b + 1].copy_(sub_ssm)
        return out

    def forward(self, x, inference_params=None, seq_idx=None, attention_mask=None):
        B, T, D = x.shape
        is_prefill = (inference_params is None) or (getattr(inference_params, "seqlen_offset", 0) == 0)

        if inference_params is not None and is_prefill:
            self._ensure_cache(inference_params, B, x.dtype)

        if inference_params is not None and is_prefill and T >= 1 and attention_mask is not None:
            am = attention_mask
            if am.size(1) > T:
                am = am[:, -T:]
            elif am.size(1) < T:
                am = None
            if am is not None:
                valid = am.bool()
                lengths = valid.long().sum(dim=1)
                if not (lengths == T).all().item():
                    if (lengths == 0).all().item():
                        return torch.zeros_like(x)
                    if T == 1:
                        nz = (lengths > 0).nonzero(as_tuple=True)[0]
                        layer_idx = getattr(self.mamba, "layer_idx", None)
                        sub_inf = _create_inference_params(
                            max_seqlen=max(getattr(inference_params, "max_seqlen", 1), 1),
                            max_batch_size=int(nz.numel()),
                            num_heads=getattr(self.mamba, "nheads", 1),
                            head_dim=getattr(self.mamba, "headdim", 64),
                            n_layer=1,
                        )
                        sub_inf.seqlen_offset = 0
                        if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                            orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                            sub_conv = orig_conv[nz].clone()
                            sub_ssm = orig_ssm[nz].clone()
                            sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
                        seq_idx_nz = seq_idx[nz].to(torch.int32).contiguous() if seq_idx is not None else None
                        out_nz = self.mamba(x[nz], inference_params=sub_inf, seq_idx=seq_idx_nz)
                        if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                            orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                            sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                            orig_conv[nz] = sub_conv
                            orig_ssm[nz] = sub_ssm
                        out = torch.zeros_like(x)
                        out[nz] = out_nz
                        self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                        return out
                    if (lengths > 0).all().item():
                        x_flat = x[valid].unsqueeze(0)
                        cu = torch.cat([
                            torch.zeros(1, device=x.device, dtype=torch.int32),
                            torch.cumsum(lengths, dim=0).to(torch.int32),
                        ])
                        seq_idx_flat = self._build_flat_seq_idx(valid, lengths, seq_idx, x.device)
                        try:
                            out_flat = self.mamba(
                                x_flat,
                                inference_params=inference_params,
                                cu_seqlens=cu,
                                seq_idx=seq_idx_flat,
                            )
                            out = torch.zeros_like(x)
                            out[valid] = out_flat.squeeze(0)
                            self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                            return out
                        except (TypeError, AssertionError):
                            out = self._run_per_sample_prefill(x, valid, lengths, inference_params, seq_idx)
                            self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                            return out
                    else:
                        nz = (lengths > 0).nonzero(as_tuple=True)[0]
                        nz_lengths = lengths[nz]
                        nz_valid = valid[nz]
                        x_flat = x[nz][nz_valid].unsqueeze(0)
                        cu = torch.cat([
                            torch.zeros(1, device=x.device, dtype=torch.int32),
                            torch.cumsum(nz_lengths, dim=0).to(torch.int32),
                        ])
                        seq_idx_nz = seq_idx[nz] if seq_idx is not None else None
                        seq_idx_flat = self._build_flat_seq_idx(nz_valid, nz_lengths, seq_idx_nz, x.device)
                        layer_idx = getattr(self.mamba, "layer_idx", None)
                        sub_inf = _create_inference_params(
                            max_seqlen=getattr(inference_params, "max_seqlen", T),
                            max_batch_size=int(nz.numel()),
                            num_heads=getattr(self.mamba, "nheads", 1),
                            head_dim=getattr(self.mamba, "headdim", 64),
                            n_layer=1,
                        )
                        sub_inf.seqlen_offset = 0
                        if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                            orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                            sub_conv = orig_conv[nz].clone()
                            sub_ssm = orig_ssm[nz].clone()
                            sub_inf.key_value_memory_dict[layer_idx] = (sub_conv, sub_ssm)
                        try:
                            out_flat = self.mamba(
                                x_flat,
                                inference_params=sub_inf,
                                cu_seqlens=cu,
                                seq_idx=seq_idx_flat,
                            )
                            if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                                orig_conv, orig_ssm = inference_params.key_value_memory_dict[layer_idx]
                                sub_conv, sub_ssm = sub_inf.key_value_memory_dict[layer_idx]
                                orig_conv[nz] = sub_conv
                                orig_ssm[nz] = sub_ssm
                            out = torch.zeros_like(x)
                            out[valid] = out_flat.squeeze(0)
                            self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                            return out
                        except (TypeError, AssertionError):
                            out = self._run_per_sample_prefill(x, valid, lengths, inference_params, seq_idx)
                            self._mask_conv_state_seq_boundary(inference_params, seq_idx, am, T)
                            return out

        if inference_params is None and attention_mask is not None and attention_mask.size(1) >= T and T > 0:
            am_bool = attention_mask[:, :T].bool()
            am_f = am_bool.unsqueeze(-1).to(x.dtype)
            x_in = x * am_f
            real_pos = am_bool.long().cumsum(dim=1) - 1
            pos_safe = torch.where(am_bool, real_pos, torch.full_like(real_pos, T))
            x_comp = torch.zeros(B, T + 1, D, device=x.device, dtype=x.dtype).scatter(
                1, pos_safe.unsqueeze(-1).expand(-1, -1, D), x_in)[:, :T, :]
            if seq_idx is not None:
                seq_comp = torch.full((B, T + 1), -1, device=seq_idx.device, dtype=torch.int32).scatter(
                    1, pos_safe, seq_idx[:, :T].to(torch.int32))[:, :T].contiguous()
                y_comp = self.mamba(x_comp, seq_idx=seq_comp)
            else:
                y_comp = self.mamba(x_comp)
            out = F.pad(y_comp, (0, 0, 0, 1)).gather(1, pos_safe.unsqueeze(-1).expand(-1, -1, D))
            return out * am_f

        if inference_params is not None:
            if not is_prefill:
                layer_idx = getattr(self.mamba, "layer_idx", None)
                if layer_idx is not None and layer_idx in inference_params.key_value_memory_dict:
                    conv_state, ssm_state = inference_params.key_value_memory_dict[layer_idx]
                    expected_dt = self._expected_step_conv_dtype(x)
                    if conv_state.dtype != expected_dt:
                        conv_tmp = conv_state.to(expected_dt)
                        inference_params.key_value_memory_dict[layer_idx] = (conv_tmp, ssm_state)
                        out = self.mamba(x, inference_params=inference_params)
                        conv_state.copy_(conv_tmp.to(conv_state.dtype))
                        inference_params.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
                        return out
                return self.mamba(x, inference_params=inference_params)
            seq_arg = seq_idx.to(torch.int32).contiguous() if seq_idx is not None else None
            out = self.mamba(x, inference_params=inference_params, seq_idx=seq_arg)
            self._mask_conv_state_seq_boundary(inference_params, seq_idx, attention_mask, T)
            return out
        if seq_idx is not None:
            return self.mamba(x, seq_idx=seq_idx.to(torch.int32).contiguous())
        return self.mamba(x)


class Engram(nn.Module):
    def __init__(self, d_model=640, num_heads=4, table_size=50000, mem_dim=64,
                 max_ngram=3):
        super().__init__()
        self.num_heads = num_heads
        self.mem_dim = mem_dim
        self.max_ngram = max_ngram
        self.tables = nn.ModuleDict()
        for n in range(2, max_ngram + 1):
            for h in range(num_heads):
                prime_size = self._next_prime(table_size + h * 997 + n * 7919)
                self.tables[f"n{n}_h{h}"] = nn.Embedding(prime_size, mem_dim)
        self.mem_proj = nn.Linear(mem_dim, d_model)
        self.gate = nn.Linear(d_model + d_model, 1)

    def _next_prime(self, n):
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x ** 0.5) + 1):
                if x % i == 0:
                    return False
            return True
        while not is_prime(n):
            n += 1
        return n

    def _poly_hash(self, ngram_ids, table_size):
        base = 31
        ids = ngram_ids.long()
        h = torch.zeros_like(ids[..., 0])
        for i in range(ids.size(-1)):
            h = (h * base + ids[..., i]) % table_size
        return h

    def forward(self, x, input_ids, T_out=None, seq_idx=None,
                attention_mask=None, is_decode=False):
        B, T_in, D = x.shape
        T_out = T_out if T_out is not None else T_in
        T_ng = input_ids.shape[1]
        use_compact = (not is_decode) and (attention_mask is not None) and (T_ng > 0)
        if use_compact:
            am_bool = attention_mask[:, :T_ng].bool()
            real_pos = am_bool.long().cumsum(dim=1) - 1
            pos_safe = torch.where(am_bool, real_pos, torch.full_like(real_pos, T_ng))
            ids_eff = torch.full((B, T_ng + 1), -1, device=input_ids.device, dtype=input_ids.dtype).scatter(
                1, pos_safe, input_ids)[:, :T_ng]
            seq_eff = (
                torch.full((B, T_ng + 1), -1, device=seq_idx.device, dtype=seq_idx.dtype).scatter(
                    1, pos_safe, seq_idx[:, :T_ng])[:, :T_ng]
                if seq_idx is not None else None
            )
            T_work = T_ng
        else:
            ids_eff = input_ids
            seq_eff = seq_idx
            T_work = T_out

        mem = torch.zeros(B, T_work, self.mem_dim, device=x.device, dtype=torch.float32)
        cnt = torch.zeros(B, T_work, 1, device=x.device, dtype=torch.float32)
        for n in range(2, self.max_ngram + 1):
            if T_ng < n:
                continue
            if is_decode:
                ngrams = ids_eff[:, -n:].unsqueeze(1)
                valid_dec = (ngrams >= 0).all(dim=-1)
                ngrams_safe = ngrams.clamp(min=0)
            else:
                ngrams = torch.stack([ids_eff[:, i:T_ng - n + 1 + i] for i in range(n)], dim=-1)
                ngrams_safe = ngrams.clamp(min=0)
                if seq_eff is not None or use_compact:
                    valid = (ngrams >= 0).all(dim=-1)
                    if seq_eff is not None:
                        ref = seq_eff[:, :T_ng - n + 1]
                        for i in range(n):
                            s_i = seq_eff[:, i:T_ng - n + 1 + i]
                            valid = valid & (s_i == ref) & (s_i >= 0)
                else:
                    valid = None
            for h in range(self.num_heads):
                table = self.tables[f"n{n}_h{h}"]
                hv = self._poly_hash(ngrams_safe, table.num_embeddings)
                if is_decode:
                    emb = table(hv).float()
                    v_f = valid_dec.to(torch.float32).unsqueeze(-1)
                    mem[:, -1:, :] += emb * v_f
                    cnt[:, -1:, :] += v_f
                else:
                    emb = table(hv).float()
                    if valid is not None:
                        v_f = valid.to(torch.float32).unsqueeze(-1)
                        mem[:, n - 1:, :] += emb * v_f
                        cnt[:, n - 1:, :] += v_f
                    else:
                        mem[:, n - 1:, :] += emb
                        cnt[:, n - 1:, :] += 1.0
        mem = mem / cnt.clamp(min=1.0)
        if use_compact:
            mem = F.pad(mem, (0, 0, 0, 1)).gather(
                1, pos_safe[:, :T_out].unsqueeze(-1).expand(-1, -1, self.mem_dim))
        mem = mem.to(x.dtype)
        mem = self.mem_proj(mem)
        gate = torch.sigmoid(self.gate(torch.cat([x, mem], dim=-1)))
        return x + gate * mem


class HybridMoELayer(nn.Module):
    def __init__(self, d_model=1024, num_t1=3, t2_rank=32, num_t2_heads=4,
                 num_prototypes=8, use_checkpoint=False, config=None):
        super().__init__()
        self.num_t1 = num_t1
        self.num_t2_heads = num_t2_heads
        self.head_rank = t2_rank // num_t2_heads
        self.use_checkpoint = use_checkpoint
        self.t1_experts = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_t1)])
        self.t1_router = nn.Linear(d_model, num_t1, bias=False)
        self.hyper_a_all = nn.Linear(d_model, num_t2_heads * d_model * self.head_rank)
        self.hyper_b_all = nn.Linear(d_model, num_t2_heads * self.head_rank * d_model)
        self.shared_expert = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.G = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.conv_query = nn.Linear(d_model, d_model, bias=False)
        self.conv_key = nn.Linear(d_model, d_model, bias=False)
        self.conv_value = nn.Linear(d_model, d_model, bias=False)
        self.conv_gate = nn.Linear(d_model, 1)
        self.conv_scale = math.sqrt(d_model)
        self.alpha = nn.Parameter(torch.full((1,), config.rezero_alpha_init if config else 0.0))
        self.gamma = nn.Parameter(torch.full((d_model,), config.layerscale_gamma_init if config else 1e-4))
        self.deepnorm_alpha = config.deepnorm_alpha if config else 1.0

    @torch._dynamo.disable
    def _forward(self, h):
        B, T, D = h.shape
        t1_outs = torch.stack([e(h) for e in self.t1_experts], dim=2)
        t1_logits = self.t1_router(h)
        t1_w = F.softmax(t1_logits, dim=-1).unsqueeze(-1)
        out_t1 = (t1_outs * t1_w).sum(dim=2)
        A_all = self.hyper_a_all(h).view(B, T, self.num_t2_heads, D, self.head_rank)
        B_all = self.hyper_b_all(h).view(B, T, self.num_t2_heads, self.head_rank, D)
        inter = torch.einsum('btd,bthdr->bthr', h, A_all)
        inter = F.gelu(inter)
        head_outs = torch.einsum('bthr,bthrd->bthd', inter, B_all)
        out_t2 = head_outs.sum(dim=2)
        cur = out_t1 + out_t2 + self.shared_expert(h)
        cur = cur * self.gamma
        q = self.conv_query(cur)
        k = self.conv_key(self.G)
        v = self.conv_value(self.G)
        attn_logits = (q @ k.t()) / self.conv_scale
        attn = torch.softmax(attn_logits, dim=-1)
        conv = attn @ v
        gate_conv = torch.sigmoid(self.conv_gate(cur))
        return self.norm(self.deepnorm_alpha * h + self.alpha * cur + gate_conv * conv)

    def forward(self, h):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, h, use_reentrant=True)
        return self._forward(h)


class LiquidCfCLayer(nn.Module):
    def __init__(self, d_model=1024, hidden_dim=16000, use_checkpoint=False, config=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.backbone = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, d_model * 3))
        self.w_tau = nn.Parameter(torch.full((d_model,), 1.0))
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.full((1,), config.rezero_alpha_init if config else 0.0))
        self.gamma = nn.Parameter(torch.full((d_model,), config.layerscale_gamma_init if config else 1e-4))
        self.deepnorm_alpha = config.deepnorm_alpha if config else 1.0

    def _forward(self, x, t):
        B, T, D = x.shape
        fgh = self.backbone(x).view(B, T, 3, D)
        f, g, h = fgh[:, :, 0], fgh[:, :, 1], fgh[:, :, 2]
        decay = F.softplus(self.w_tau) * torch.sigmoid(f)
        gate = torch.exp(-decay * t)
        out = gate * g + (1 - gate) * h
        out = out * self.gamma
        return self.norm(self.deepnorm_alpha * x + self.alpha * out)

    def forward(self, x, t=1.0):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, t, use_reentrant=True)
        return self._forward(x, t)


class MultiDimLayer(nn.Module):
    def __init__(self, d_model=1024, base_dim=2048, curvature=1.0,
                 use_checkpoint=False, config=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.base_dim = base_dim
        self.c = curvature
        self.sqrt_c = curvature ** 0.5
        self.eps = 1e-7
        self.euclid = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(),
                                     nn.Linear(1024, base_dim))
        self.hyper = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(),
                                    nn.Linear(1024, base_dim))
        self.sph = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(),
                                  nn.Linear(1024, base_dim))
        self.to_ball = nn.Linear(d_model, base_dim)
        self.from_ball = nn.Linear(base_dim, d_model)
        self.hyper_shift_raw = nn.Parameter(torch.zeros(base_dim))
        self.t_embed = nn.Embedding(32, 16)
        self.gate = nn.Sequential(nn.Linear(base_dim * 3 + 16, 1024), nn.GELU(),
                                    nn.Linear(1024, 3), nn.Softmax(dim=-1))
        self.out_proj = nn.Linear(base_dim * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.full((1,), config.rezero_alpha_init if config else 0.0))
        self.gamma = nn.Parameter(torch.full((d_model,), config.layerscale_gamma_init if config else 1e-4))
        self.deepnorm_alpha = config.deepnorm_alpha if config else 1.0

    def _safe_norm(self, v):
        return (v * v).sum(dim=-1, keepdim=True).clamp(min=self.eps * self.eps).sqrt()

    def _project_to_ball(self, v, max_norm=0.95):
        limit = max_norm / self.sqrt_c
        n = self._safe_norm(v)
        scale = torch.clamp(limit / n, max=1.0)
        return v * scale

    def _exp_0(self, v):
        n = self._safe_norm(v)
        scale = torch.tanh(self.sqrt_c * n) / (self.sqrt_c * n)
        return v * scale

    def _log_0(self, y):
        n = self._safe_norm(y)
        n_safe = n.clamp(min=self.eps, max=1.0 / self.sqrt_c - 1e-4)
        scale = torch.atanh(self.sqrt_c * n_safe) / (self.sqrt_c * n_safe)
        return y * scale

    def _mobius_add(self, x, y):
        c = self.c
        x2 = (x * x).sum(-1, keepdim=True).clamp(max=1.0 / c - 1e-4)
        y2 = (y * y).sum(-1, keepdim=True).clamp(max=1.0 / c - 1e-4)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = (1 + 2 * c * xy + c * c * x2 * y2).clamp(min=self.eps)
        return self._project_to_ball(num / den)

    def _lambda_x(self, x):
        x2 = (x * x).sum(-1, keepdim=True).clamp(max=1.0 / self.c - 1e-4)
        return 2.0 / (1.0 - self.c * x2).clamp(min=self.eps)

    def _exp_x(self, x, v):
        lam = self._lambda_x(x)
        v_norm = self._safe_norm(v)
        scale = torch.tanh(self.sqrt_c * lam * v_norm / 2.0) / (self.sqrt_c * v_norm)
        return self._mobius_add(x, v * scale)

    def _parallel_transport_0_to_x(self, v, x):
        x2 = (x * x).sum(-1, keepdim=True).clamp(max=1.0 / self.c - 1e-4)
        return (1.0 - self.c * x2) * v

    def _forward(self, h, t_idx):
        e_raw = self.euclid(h)
        hyp_delta_raw = self.hyper(h)
        sph_raw = self.sph(h)
        h_ball = self._exp_0(self.to_ball(h).float())
        hyp_delta = self._exp_0(hyp_delta_raw.float())
        hyper_shift = self._exp_0(self.hyper_shift_raw.float())
        hyp_ball = self._mobius_add(h_ball, hyp_delta)
        hyp_ball = self._mobius_add(hyp_ball, hyper_shift.unsqueeze(0).unsqueeze(0))
        hyp_tan = self._log_0(hyp_ball).to(h.dtype)
        sph = F.normalize(sph_raw.float(), dim=-1).to(h.dtype)
        B, T, _ = h.shape
        if t_idx.dim() == 0:
            t_idx = t_idx.unsqueeze(0)
        if t_idx.numel() == 1:
            t_vec = self.t_embed(t_idx).squeeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        elif t_idx.numel() == B:
            t_vec = self.t_embed(t_idx).unsqueeze(1).expand(-1, T, -1)
        else:
            t_vec = self.t_embed(t_idx[:1]).squeeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        comb = torch.cat([e_raw, hyp_tan, sph, t_vec], dim=-1)
        gw = self.gate(comb)
        gated = torch.cat([e_raw * gw[:, :, 0:1], hyp_tan * gw[:, :, 1:2],
                           sph * gw[:, :, 2:3]], dim=-1)
        out = self.out_proj(gated) * self.gamma
        delta = self.alpha * out
        delta_ball = self.to_ball(delta).float()
        delta_pt = self._parallel_transport_0_to_x(delta_ball, h_ball)
        h_new_ball = self._exp_x(h_ball, delta_pt)
        h_new = self.from_ball(self._log_0(h_new_ball).to(h.dtype))
        return self.norm(h_new)

    def forward(self, h, t_idx=None):
        if t_idx is None:
            t_idx = torch.zeros(1, device=h.device, dtype=torch.long)
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, h, t_idx, use_reentrant=True)
        return self._forward(h, t_idx)


class LightThought(nn.Module):
    def __init__(self, d_model=1024, ratio=4):
        super().__init__()
        hidden = d_model * ratio
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.down(F.silu(self.gate(x)) * self.up(x)))


class HyperConnection(nn.Module):
    def __init__(self, num_streams=16, d_model=640, top_k=4):
        super().__init__()
        self.num_streams = num_streams
        self.top_k = min(top_k, num_streams)
        self.in_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_streams)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_streams)])
        with torch.no_grad():
            for i in range(num_streams):
                nn.init.eye_(self.in_proj[i].weight)
                nn.init.eye_(self.out_proj[i].weight)
                if i > 0:
                    self.in_proj[i].weight.data += torch.randn_like(self.in_proj[i].weight) * 0.02
                    self.out_proj[i].weight.data += torch.randn_like(self.out_proj[i].weight) * 0.02
        self.stream_weights = nn.Parameter(torch.zeros(num_streams))
        hidden_router = max(d_model // 4, 64)
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, hidden_router),
            nn.SiLU(),
            nn.Linear(hidden_router, 1),
        )

    def forward(self, streams, layer_fn):
        n = self.num_streams
        w = F.softmax(self.stream_weights, dim=0)
        stacked = torch.stack(streams, dim=1)
        B, _, T, D = stacked.shape
        w_cast = w.to(stacked.dtype)
        W_in = torch.stack([p.weight for p in self.in_proj], dim=0).to(stacked.dtype)
        w_streams = (stacked * w_cast.view(1, n, 1, 1)).permute(1, 0, 2, 3).reshape(n, B * T, D)
        x_in = torch.bmm(w_streams, W_in.transpose(-1, -2)).sum(dim=0).view(B, T, D)
        x_out = layer_fn(x_in)
        x_out_e = x_out.unsqueeze(1).expand(-1, n, -1, -1)
        combined = torch.cat([stacked, x_out_e], dim=-1)
        scores = self.router(combined).mean(dim=(2, 3))
        topk_idx = scores.topk(self.top_k, dim=-1).indices
        W_out = torch.stack([p.weight for p in self.out_proj], dim=0).to(x_out.dtype)
        W_sel = W_out[topk_idx]
        deltas_sel = torch.matmul(x_out.unsqueeze(1), W_sel.transpose(-1, -2))
        new_stacked = stacked.scatter_add(
            1,
            topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D),
            deltas_sel,
        )
        return [new_stacked[:, i] for i in range(n)]


class BlockRouter:
    def __init__(self, top_k_a=4, top_k_b=4):
        self.top_k_a = top_k_a
        self.top_k_b = top_k_b

    def __call__(self, C, block_size, num_blocks, attention_mask=None):
        B, T = C.shape
        pad = num_blocks * block_size - T
        C_p = F.pad(C.float(), (0, pad), value=0.0) if pad > 0 else C.float()
        C_blocks = C_p.view(B, num_blocks, block_size)
        if attention_mask is not None:
            am = attention_mask[:, :T].float()
        else:
            am = torch.ones(B, T, device=C.device, dtype=torch.float32)
        if pad > 0:
            am = F.pad(am, (0, pad), value=0.0)
        am_blocks = am.view(B, num_blocks, block_size)
        raw_counts = am_blocks.sum(-1)
        valid = raw_counts > 0
        counts = raw_counts.clamp(min=1.0)
        C_masked = C_blocks * am_blocks
        score_a = C_masked.max(dim=-1).values
        score_b = C_masked.sum(dim=-1) / counts
        score_a = score_a.masked_fill(~valid, float('-inf'))
        score_b = score_b.masked_fill(~valid, float('-inf'))
        k_a = min(self.top_k_a, num_blocks)
        k_b = min(self.top_k_b, num_blocks)
        _, idx_a = score_a.topk(k_a, dim=-1)
        _, idx_b = score_b.topk(k_b, dim=-1)
        mask_a = torch.zeros(B, num_blocks, dtype=torch.bool, device=C.device)
        mask_b = torch.zeros(B, num_blocks, dtype=torch.bool, device=C.device)
        mask_a.scatter_(1, idx_a, True)
        mask_b.scatter_(1, idx_b, True)
        selected = (mask_a | mask_b) & valid
        advanced = (mask_a & mask_b) & valid
        return selected, advanced

    def route_packed(self, C, block_size, num_blocks, seq_idx, num_seqs, attention_mask=None):
        B, T = C.shape
        device = C.device
        pad = num_blocks * block_size - T
        C_p = F.pad(C.float(), (0, pad), value=0.0) if pad > 0 else C.float()
        seq_p = F.pad(seq_idx[:, :T].to(torch.int32), (0, pad), value=-1) if pad > 0 else seq_idx[:, :T].to(torch.int32)
        if attention_mask is not None:
            am = attention_mask[:, :T].float()
            if pad > 0:
                am = F.pad(am, (0, pad), value=0.0)
        else:
            am = torch.ones(B, T, device=device, dtype=torch.float32)
            if pad > 0:
                am = F.pad(am, (0, pad), value=0.0)
        C_blocks = C_p.view(B, 1, num_blocks, block_size)
        seq_blocks = seq_p.view(B, 1, num_blocks, block_size)
        am_blocks = (am.view(B, 1, num_blocks, block_size) > 0)
        seq_range = torch.arange(1, num_seqs + 1, device=device, dtype=torch.int32).view(1, num_seqs, 1, 1)
        m_s = (seq_blocks == seq_range) & am_blocks
        m_sf = m_s.float()
        cnt_s = m_sf.sum(-1)
        valid_s = cnt_s > 0
        C_ms = C_blocks * m_sf
        score_a = C_ms.max(dim=-1).values.masked_fill(~valid_s, float('-inf'))
        score_b = (C_ms.sum(dim=-1) / cnt_s.clamp(min=1.0)).masked_fill(~valid_s, float('-inf'))
        k_a = min(self.top_k_a, num_blocks)
        k_b = min(self.top_k_b, num_blocks)
        _, idx_a = score_a.topk(k_a, dim=-1)
        _, idx_b = score_b.topk(k_b, dim=-1)
        mask_a = torch.zeros(B, num_seqs, num_blocks, dtype=torch.bool, device=device)
        mask_b = torch.zeros(B, num_seqs, num_blocks, dtype=torch.bool, device=device)
        mask_a.scatter_(2, idx_a, True)
        mask_b.scatter_(2, idx_b, True)
        selected_s = (mask_a | mask_b) & valid_s
        advanced_s = (mask_a & mask_b) & valid_s
        return selected_s, advanced_s


class GlobalBlock(nn.Module):
    def __init__(self, d_model, d_model_thought, num_heads, num_kv_heads, head_dim,
                 window_size, use_attention_checkpoint=False):
        super().__init__()
        self.use_attention_checkpoint = use_attention_checkpoint
        self.conv = CausalConv1d(d_model)
        self.attn = GQAAttention(d_model, num_heads, num_kv_heads, head_dim, window_size)
        self.up_proj = nn.Linear(d_model, d_model_thought)
        self.ffn = LightThought(d_model_thought)
        self.down_proj = nn.Linear(d_model_thought, d_model)

    def forward(self, x, past_kv=None, conv_cache=None, seq_idx=None,
                attention_mask=None, return_importance=False,
                conv_seq_cache=None, past_seq_idx=None, return_seq_cache=False,
                q_pos=None, past_kv_pos=None):
        if return_seq_cache:
            x_conv, new_conv, new_conv_seq = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=True)
        else:
            x_conv, new_conv = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=False)
            new_conv_seq = None
        if self.use_attention_checkpoint and self.training and torch.is_grad_enabled() and past_kv is None:
            if return_importance:
                def _attn_fn(inp):
                    out, _, c_imp = self.attn._core(
                        inp, None, seq_idx, attention_mask, True, None, q_pos, None)
                    return out, c_imp
                x_attn, C = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                new_kv = None
            else:
                def _attn_fn(inp):
                    return self.attn._core(
                        inp, None, seq_idx, attention_mask, False, None, q_pos, None)[0]
                x_attn = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                new_kv = None
                C = None
        else:
            x_attn, new_kv, C = self.attn(x_conv, past_kv, seq_idx,
                                           attention_mask=attention_mask,
                                           return_importance=return_importance,
                                           past_seq_idx=past_seq_idx,
                                           q_pos=q_pos,
                                           past_kv_pos=past_kv_pos)
        x_up = self.up_proj(x_attn)
        x_ffn = self.ffn(x_up)
        x_out = self.down_proj(x_ffn)
        if return_seq_cache:
            return x_out, new_kv, new_conv, C, new_conv_seq
        return x_out, new_kv, new_conv, C


class ThoughtBlock(nn.Module):
    def __init__(self, d_model, d_model_thought, num_heads, num_kv_heads, head_dim,
                 block_size, top_k_heads, thought_layer, max_blocks, max_full,
                 mamba_d_state=128, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64,
                 ssm_cache_dtype=None, conv_cache_dtype=None,
                 use_attention_checkpoint=False):
        super().__init__()
        self.use_attention_checkpoint = use_attention_checkpoint
        self.conv = CausalConv1d(d_model)
        self.attn = ThoughtGQAAttention(d_model, num_heads, num_kv_heads, head_dim,
                                          block_size, top_k_heads, max_blocks, max_full)
        self.up_proj = nn.Linear(d_model, d_model_thought)
        self.mamba2 = Mamba2Layer(d_model_thought, d_state=mamba_d_state,
                                    d_conv=mamba_d_conv, expand=mamba_expand,
                                    headdim=mamba_headdim,
                                    ssm_cache_dtype=ssm_cache_dtype,
                                    conv_cache_dtype=conv_cache_dtype)
        self.mamba_gate = nn.Parameter(torch.zeros(1))
        self.thought = thought_layer
        self.down_proj = nn.Linear(d_model_thought, d_model)

    def forward(self, x, cache=None, conv_cache=None, seq_idx=None, t=1.0, t_idx=None,
                attention_mask=None, selected=None, advanced=None, reeval=False,
                build_cache=False, mamba_inference_params=None,
                conv_seq_cache=None, cur_seq_ids=None, return_seq_cache=False,
                real_pos=None, selected_per_seq=None, advanced_per_seq=None, reeval_mask=None):
        if return_seq_cache:
            x_conv, new_conv, new_conv_seq = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=True)
        else:
            x_conv, new_conv = self.conv(
                x, conv_cache, seq_idx=seq_idx, attention_mask=attention_mask,
                seq_cache=conv_seq_cache, return_seq_cache=False)
            new_conv_seq = None
        if cache is None:
            if self.use_attention_checkpoint and self.training and torch.is_grad_enabled() and not build_cache:
                def _attn_fn(inp):
                    return self.attn._core_prefill(
                        inp, seq_idx, attention_mask, selected, advanced, False,
                        real_pos=real_pos, selected_per_seq=selected_per_seq,
                        advanced_per_seq=advanced_per_seq)[0]
                x_attn = checkpoint(_attn_fn, x_conv, use_reentrant=True)
                new_attn_cache = None
            else:
                x_attn, new_attn_cache = self.attn.forward_prefill(
                    x_conv, seq_idx=seq_idx, attention_mask=attention_mask,
                    selected=selected, advanced=advanced, build_cache=build_cache,
                    real_pos=real_pos, selected_per_seq=selected_per_seq,
                    advanced_per_seq=advanced_per_seq)
        else:
            x_attn, new_attn_cache = self.attn.forward_decode(
                x_conv, cache, selected=selected, advanced=advanced,
                reeval=reeval, cur_seq_ids=cur_seq_ids, reeval_mask=reeval_mask)
        x_up = self.up_proj(x_attn)

        x_mamba = self.mamba2(x_up, inference_params=mamba_inference_params,
                              seq_idx=seq_idx, attention_mask=attention_mask)
        gate = torch.sigmoid(self.mamba_gate)
        x_comb = x_up + gate * x_mamba

        if isinstance(self.thought, LiquidCfCLayer):
            x_th = self.thought(x_comb, t=t)
        elif isinstance(self.thought, MultiDimLayer):
            x_th = self.thought(x_comb, t_idx=t_idx)
        else:
            x_th = self.thought(x_comb)
        x_out = self.down_proj(x_th)
        if return_seq_cache:
            return x_out, new_attn_cache, new_conv, new_conv_seq
        return x_out, new_attn_cache, new_conv


class CouncilBackbone(nn.Module):
    def __init__(self, d_model=640, d_model_thought=1024, num_blocks=7, num_streams=16,
                 vocab_size=50257, use_checkpoint=False, window_size=4096,
                 attention_checkpoint=None, config=None):
        super().__init__()
        self.d_model = d_model
        self.d_model_thought = d_model_thought
        self.num_blocks = num_blocks
        self.num_streams = num_streams
        self.use_checkpoint = use_checkpoint
        self.attention_checkpoint = attention_checkpoint if attention_checkpoint is not None else use_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.engram = Engram(d_model, table_size=config.engram_table_size,
                              mem_dim=config.engram_mem_dim)
        self.block_size = config.sliding_window // config.sparse_num_blocks
        self.max_blocks = (config.max_seq_len + self.block_size - 1) // self.block_size
        self.max_full = config.adv_top_k_a + config.adv_top_k_b + config.hybrid_max_full_extra

        g_heads = config.global_num_heads
        g_kv = config.global_num_kv_heads
        t_heads = config.thought_num_heads
        t_kv = config.thought_num_kv_heads
        hd = config.head_dim

        self.blocks = nn.ModuleList()
        mamba_layer_count = 0
        ssm_cache_dtype = getattr(config, "mamba_ssm_cache_dtype", None)
        conv_cache_dtype = getattr(config, "mamba_conv_cache_dtype", None)
        for i in range(num_blocks):
            if i % 2 == 0:
                block = GlobalBlock(d_model, d_model_thought, g_heads, g_kv, hd, window_size,
                                    use_attention_checkpoint=self.attention_checkpoint)
            else:
                if i == 1:
                    th = HybridMoELayer(d_model_thought,
                                        num_prototypes=getattr(config, "moe_num_prototypes", 8),
                                        use_checkpoint=use_checkpoint, config=config)
                elif i == 3:
                    th = LiquidCfCLayer(d_model_thought, config.dynamic_hidden,
                                         use_checkpoint=use_checkpoint, config=config)
                elif i == 5:
                    th = MultiDimLayer(d_model_thought, config.multidim_base,
                                        use_checkpoint=use_checkpoint, config=config)
                else:
                    th = LightThought(d_model_thought)
                block = ThoughtBlock(d_model, d_model_thought, t_heads, t_kv, hd,
                                      self.block_size, config.thought_head_top_k, th,
                                      self.max_blocks, self.max_full,
                                      mamba_d_state=config.mamba_d_state,
                                      mamba_d_conv=config.mamba_d_conv,
                                      mamba_expand=config.mamba_expand,
                                      mamba_headdim=config.mamba_headdim,
                                      ssm_cache_dtype=ssm_cache_dtype,
                                      conv_cache_dtype=conv_cache_dtype,
                                      use_attention_checkpoint=self.attention_checkpoint)
                block.mamba2.mamba.layer_idx = mamba_layer_count
                mamba_layer_count += 1
            self.blocks.append(block)

        self.hcs = nn.ModuleList([HyperConnection(num_streams, d_model, top_k=config.hyper_top_k)
                                    for _ in range(num_blocks)])
        self.final_mix = nn.Parameter(torch.zeros(num_streams))
        self.norm = nn.LayerNorm(d_model)
        self.block_router = BlockRouter(top_k_a=config.adv_top_k_a, top_k_b=config.adv_top_k_b)
        self.reeval_interval = getattr(config, "hybrid_reeval_interval", 128)
        self.max_seqlen = config.max_seq_len
        self.mamba_layer_count = mamba_layer_count
        self.mamba_nheads = (d_model_thought * config.mamba_expand) // config.mamba_headdim
        self.config_mamba_headdim = config.mamba_headdim

    def _build_seq_idx_from_cu_seqlens(self, cu_seqlens, T, attention_mask, device, target_batch=None):
        if cu_seqlens.dim() == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        B_cu, _ = cu_seqlens.shape
        seq_idx = torch.zeros(B_cu, T, dtype=torch.int32, device=device)
        positions = torch.arange(T, device=device)
        for b in range(B_cu):
            row = cu_seqlens[b].to(torch.long)
            valid_vals = row[row > 0]
            if valid_vals.numel() > 0:
                max_end = int(valid_vals.max().item())
                boundaries = valid_vals[valid_vals < min(max_end, T)]
                if boundaries.numel() > 0:
                    boundaries = torch.unique_consecutive(boundaries)
                    counts = (positions.unsqueeze(0) >= boundaries.unsqueeze(1)).sum(0)
                else:
                    counts = torch.zeros(T, dtype=torch.long, device=device)
                row_seq = (1 + counts).to(torch.int32)
                if max_end < T:
                    row_seq = torch.where(positions < max_end, row_seq, torch.full_like(row_seq, -1))
            elif (row >= 0).any() and row.numel() >= 2:
                row_seq = torch.full((T,), -1, dtype=torch.int32, device=device)
            else:
                row_seq = torch.ones(T, dtype=torch.int32, device=device)
            if attention_mask is not None and attention_mask.size(0) == B_cu and attention_mask.size(1) >= T:
                row_seq = torch.where(attention_mask[b, :T].bool(), row_seq, torch.full_like(row_seq, -1))
            seq_idx[b] = row_seq
        if target_batch is not None:
            if B_cu == 1 and target_batch > 1:
                seq_idx = seq_idx.expand(target_batch, -1).clone()
                if attention_mask is not None and attention_mask.size(0) == target_batch and attention_mask.size(1) >= T:
                    seq_idx = torch.where(attention_mask[:, :T].bool(), seq_idx, torch.full_like(seq_idx, -1))
            elif B_cu != target_batch:
                raise ValueError(f"cu_seqlens batch size {B_cu} does not match input batch size {target_batch}")
        return seq_idx

    def _extract_active_seq_ids(self, seq_idx, T):
        B = seq_idx.shape[0]
        device = seq_idx.device
        seq_t = seq_idx[:, :T].to(torch.int32)
        pos_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid_seq = seq_t >= 0
        last_valid_pos = torch.where(valid_seq, pos_grid, torch.full_like(pos_grid, -1)).max(dim=1).values.clamp(min=0)
        active = seq_t[torch.arange(B, device=device), last_valid_pos]
        return torch.where(valid_seq.any(dim=1), active, torch.full_like(active, -1))

    def _compact_importance_and_mask(self, C, kv_pos, num_blocks, seq_vals=None, target_seq=None):
        B = C.shape[0]
        max_len = num_blocks * self.block_size
        C_log = torch.zeros(B, max_len, device=C.device, dtype=torch.float32)
        am_log = torch.zeros(B, max_len, device=C.device, dtype=torch.float32)
        seq_log = torch.full((B, max_len), -1, device=C.device, dtype=torch.int32) if seq_vals is not None else None
        if max_len <= 0 or kv_pos.size(1) == 0:
            return (C_log, am_log, seq_log) if seq_vals is not None else (C_log, am_log)
        valid = (kv_pos >= 0) & (kv_pos < max_len)
        if seq_vals is not None and target_seq is not None:
            valid = valid & (seq_vals == target_seq.view(B, 1)) & (target_seq.view(B, 1) >= 0)
        pos_safe = kv_pos.clamp(min=0, max=max_len - 1)
        v_f = valid.float()
        C_log = C_log.scatter_add(1, pos_safe, C.float() * v_f)
        am_log = am_log.scatter_add(1, pos_safe, v_f)
        if seq_vals is not None:
            seq_in = torch.where(valid, seq_vals.to(torch.int32), torch.full_like(seq_vals, -1, dtype=torch.int32))
            seq_log = torch.full((B, max_len + 1), -1, device=C.device, dtype=torch.int32).scatter(
                1, torch.where(valid, pos_safe, torch.full_like(pos_safe, max_len)), seq_in)[:, :max_len]
            return C_log, am_log, seq_log
        return C_log, am_log

    def _align_route_width(self, sel, adv, target_nb):
        cur_nb = sel.shape[1]
        if cur_nb >= target_nb:
            return sel[:, :target_nb], adv[:, :target_nb]
        pad = target_nb - cur_nb
        return F.pad(sel, (0, pad), value=False), F.pad(adv, (0, pad), value=False)

    def forward(self, input_ids, cache=None, return_cache=False,
                cu_seqlens=None, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        is_decode = cache is not None
        want_cache = return_cache and not self.training

        seq_idx = None
        active_seq_ids = None
        num_packed_seqs = 0
        if is_decode:
            active_seq_ids = cache.get("active_seq_ids", None)
            if active_seq_ids is not None:
                seq_idx = active_seq_ids.view(B, 1)
            attn_seq_idxs = list(cache.get("attn_seq_idxs", [None] * self.num_blocks))
            conv_seq_caches = list(cache.get("conv_seq_caches", [None] * self.num_blocks))
        else:
            if cu_seqlens is not None:
                seq_idx = self._build_seq_idx_from_cu_seqlens(cu_seqlens, T, attention_mask, device, target_batch=B)
                active_seq_ids = self._extract_active_seq_ids(seq_idx, T)
                num_packed_seqs = int(seq_idx.max().item()) if seq_idx.numel() > 0 else 0
            attn_seq_idxs = [None] * self.num_blocks
            conv_seq_caches = [None] * self.num_blocks

        if is_decode:
            input_ids_last = input_ids[:, -1:]
            x = self.embedding(input_ids_last)
            if "last_tokens" in cache and cache["last_tokens"] is not None:
                toks = torch.cat([cache["last_tokens"], input_ids_last], dim=1)
                x = self.engram(x, toks, T_out=x.shape[1], is_decode=True)
            attn_kvs = list(cache["attn_kvs"])
            conv_caches = list(cache["conv_caches"])
            thought_caches = list(cache["thought_caches"])
            mamba_inf = cache.get("mamba_inference_params", None)
            streams = [x.clone() for _ in range(self.num_streams)]
            past_seq_len = cache.get("seq_len", 0)
            T_step = x.shape[1]
            row_seq_lens_before = cache.get(
                "row_seq_lens",
                torch.full((B,), past_seq_len, device=device, dtype=torch.long),
            )
            cur_q_pos = row_seq_lens_before.unsqueeze(1) + torch.arange(T_step, device=device).unsqueeze(0)
            past_kv_pos = cache.get("attn_kv_pos", None)
            if past_kv_pos is None:
                if past_seq_len > 0:
                    past_kv_pos = torch.arange(past_seq_len, device=device).unsqueeze(0).expand(B, past_seq_len)
                else:
                    past_kv_pos = torch.zeros(B, 0, device=device, dtype=torch.long)
            full_kv_pos = torch.cat([past_kv_pos, cur_q_pos], dim=1)
            row_seq_lens_after = row_seq_lens_before + T_step
            real_pos = None
        else:
            x = self.embedding(input_ids)
            x = self.engram(x, input_ids, seq_idx=seq_idx,
                              attention_mask=attention_mask, is_decode=False)
            attn_kvs = [None] * self.num_blocks
            conv_caches = [None] * self.num_blocks
            thought_caches = [None] * self.num_blocks
            if want_cache:
                mamba_inf = _create_inference_params(
                    max_seqlen=max(self.max_seqlen, T), max_batch_size=B,
                    num_heads=self.mamba_nheads,
                    head_dim=self.config_mamba_headdim if hasattr(self, "config_mamba_headdim") else 64,
                    n_layer=self.mamba_layer_count,
                )
                mamba_inf.seqlen_offset = 0
            else:
                mamba_inf = None
            streams = [x.clone() for _ in range(self.num_streams)]
            past_seq_len = 0
            T_step = T
            if attention_mask is not None:
                valid_tok = attention_mask[:, :T].bool()
                real_pos = torch.where(
                    valid_tok,
                    valid_tok.long().cumsum(dim=1) - 1,
                    torch.full((B, T), -1, device=device, dtype=torch.long),
                )
                row_seq_lens_after = valid_tok.long().sum(dim=1)
                full_kv_pos = real_pos
            else:
                real_pos = None
                row_seq_lens_after = torch.full((B,), T, device=device, dtype=torch.long)
                full_kv_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T) if want_cache else None
            cur_q_pos = real_pos
            past_kv_pos = None

        if is_decode:
            max_row_len = int(row_seq_lens_after.max().item()) if B > 0 else (past_seq_len + T_step)
            num_attn_blocks = max(self.max_blocks, (max_row_len + self.block_size - 1) // self.block_size)
        else:
            num_attn_blocks = max((T_step + self.block_size - 1) // self.block_size, self.max_blocks)

        if is_decode:
            saved_routes = cache.get("router_routes", None)
            if saved_routes is not None and len(saved_routes) == self.num_blocks:
                router_outputs = list(saved_routes)
            else:
                router_outputs = [None] * self.num_blocks
                ls = cache.get("last_sel", None)
                la = cache.get("last_adv", None)
                if ls is not None and la is not None:
                    for j in range(self.num_blocks - 1):
                        if isinstance(self.blocks[j], GlobalBlock):
                            router_outputs[j] = (ls, la)
            if self.reeval_interval > 0:
                reeval_rows = (row_seq_lens_before > 0) & ((row_seq_lens_before % self.reeval_interval) == 0)
            else:
                reeval_rows = torch.zeros(B, dtype=torch.bool, device=device)
            reeval_now = bool(reeval_rows.any().item())
        else:
            router_outputs = [None] * self.num_blocks
            reeval_rows = None
            reeval_now = False

        router_per_seq = [None] * self.num_blocks
        track_seq_cache = (is_decode or want_cache) and (seq_idx is not None)

        for i in range(self.num_blocks):
            block = self.blocks[i]
            is_global = isinstance(block, GlobalBlock)
            is_last_global = is_global and (i == self.num_blocks - 1)

            if not is_global:
                sel_adv = None
                sel_adv_seq = None
                for k in range(i - 1, -1, -1):
                    if router_outputs[k] is not None:
                        sel_k, adv_k = router_outputs[k]
                        sel_adv = self._align_route_width(sel_k, adv_k, num_attn_blocks)
                        if router_per_seq[k] is not None:
                            sel_adv_seq = router_per_seq[k]
                        break
                if sel_adv is None:
                    B_cur = streams[0].shape[0]
                    sel_d = torch.ones(B_cur, num_attn_blocks, dtype=torch.bool, device=device)
                    adv_d = torch.zeros_like(sel_d)
                    sel_adv = (sel_d, adv_d)
            else:
                sel_adv = None
                sel_adv_seq = None

            def make_layer_fn(i, block, is_global, is_last_global, sel_adv, sel_adv_seq, router_outputs, router_per_seq):
                def layer_fn(h):
                    nonlocal attn_kvs, conv_caches, thought_caches, mamba_inf, attn_seq_idxs, conv_seq_caches
                    t_val = float(i + 1)
                    t_idx = torch.tensor([i % 32], device=device, dtype=torch.long)
                    if is_global:
                        need_router = not is_last_global
                        force_c = (not is_decode and need_router) or (is_decode and reeval_now and need_router)
                        past_s = attn_seq_idxs[i] if is_decode else None
                        if track_seq_cache:
                            h_out, new_kv, new_conv, C, new_conv_seq = block(
                                h, past_kv=attn_kvs[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx, attention_mask=attention_mask,
                                return_importance=force_c,
                                conv_seq_cache=conv_seq_caches[i],
                                past_seq_idx=past_s,
                                return_seq_cache=True,
                                q_pos=cur_q_pos,
                                past_kv_pos=past_kv_pos)
                        else:
                            h_out, new_kv, new_conv, C = block(
                                h, past_kv=attn_kvs[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx, attention_mask=attention_mask,
                                return_importance=force_c,
                                q_pos=cur_q_pos,
                                past_kv_pos=past_kv_pos)
                            new_conv_seq = None
                        if need_router and C is not None:
                            if not is_decode and seq_idx is not None and num_packed_seqs > 1:
                                kv_pos_eff = full_kv_pos if full_kv_pos is not None else torch.arange(C.size(1), device=device).unsqueeze(0).expand(B, C.size(1))
                                C_log, am_log, seq_log = self._compact_importance_and_mask(
                                    C, kv_pos_eff, num_attn_blocks, seq_vals=seq_idx[:, :C.size(1)])
                                sel_s, adv_s = self.block_router.route_packed(
                                    C_log, self.block_size, num_attn_blocks, seq_log,
                                    num_packed_seqs, attention_mask=am_log)
                                router_per_seq[i] = (sel_s, adv_s)
                                b_all_r = torch.arange(B, device=device)
                                active_idx = (active_seq_ids - 1).clamp(min=0, max=num_packed_seqs - 1)
                                has_act = active_seq_ids >= 1
                                sel = torch.where(has_act.unsqueeze(1), sel_s[b_all_r, active_idx], torch.zeros(B, num_attn_blocks, dtype=torch.bool, device=device))
                                adv = torch.where(has_act.unsqueeze(1), adv_s[b_all_r, active_idx], torch.zeros(B, num_attn_blocks, dtype=torch.bool, device=device))
                                router_outputs[i] = (sel, adv)
                            elif full_kv_pos is not None:
                                full_s_vals = torch.cat([past_s, seq_idx[:, :T_step]], dim=1) if (is_decode and past_s is not None and seq_idx is not None) else (seq_idx[:, :C.size(1)] if seq_idx is not None else None)
                                if full_s_vals is not None and active_seq_ids is not None:
                                    C_log, am_log, _ = self._compact_importance_and_mask(
                                        C, full_kv_pos, num_attn_blocks, seq_vals=full_s_vals, target_seq=active_seq_ids)
                                else:
                                    C_log, am_log = self._compact_importance_and_mask(C, full_kv_pos, num_attn_blocks)
                                new_sel, new_adv = self.block_router(
                                    C_log, self.block_size, num_attn_blocks,
                                    attention_mask=am_log)
                                if is_decode and router_outputs[i] is not None:
                                    old_sel, old_adv = self._align_route_width(
                                        router_outputs[i][0], router_outputs[i][1], num_attn_blocks)
                                    r_col = reeval_rows.unsqueeze(1)
                                    new_sel = torch.where(r_col, new_sel, old_sel)
                                    new_adv = torch.where(r_col, new_adv, old_adv)
                                router_outputs[i] = (new_sel, new_adv)
                            else:
                                new_sel, new_adv = self.block_router(
                                    C, self.block_size, num_attn_blocks,
                                    attention_mask=attention_mask)
                                if is_decode and router_outputs[i] is not None:
                                    old_sel, old_adv = self._align_route_width(
                                        router_outputs[i][0], router_outputs[i][1], num_attn_blocks)
                                    r_col = reeval_rows.unsqueeze(1)
                                    new_sel = torch.where(r_col, new_sel, old_sel)
                                    new_adv = torch.where(r_col, new_adv, old_adv)
                                router_outputs[i] = (new_sel, new_adv)
                        if is_decode or want_cache:
                            attn_kvs[i] = new_kv
                            conv_caches[i] = new_conv
                            if track_seq_cache:
                                conv_seq_caches[i] = new_conv_seq
                                if is_decode and past_s is not None:
                                    attn_seq_idxs[i] = torch.cat([past_s, seq_idx[:, :T_step]], dim=1)
                                elif not is_decode:
                                    attn_seq_idxs[i] = seq_idx[:, :T_step]
                        return h_out
                    else:
                        sel, adv = sel_adv
                        sel_ps, adv_ps = sel_adv_seq if sel_adv_seq is not None else (None, None)
                        if track_seq_cache:
                            h_out, new_attn_cache, new_conv, new_conv_seq = block(
                                h, cache=thought_caches[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx, t=t_val, t_idx=t_idx,
                                attention_mask=attention_mask,
                                selected=sel, advanced=adv,
                                reeval=reeval_now,
                                build_cache=(want_cache and not is_decode),
                                mamba_inference_params=mamba_inf,
                                conv_seq_cache=conv_seq_caches[i],
                                cur_seq_ids=active_seq_ids,
                                return_seq_cache=True,
                                real_pos=real_pos,
                                selected_per_seq=sel_ps,
                                advanced_per_seq=adv_ps,
                                reeval_mask=reeval_rows)
                        else:
                            h_out, new_attn_cache, new_conv = block(
                                h, cache=thought_caches[i], conv_cache=conv_caches[i],
                                seq_idx=seq_idx, t=t_val, t_idx=t_idx,
                                attention_mask=attention_mask,
                                selected=sel, advanced=adv,
                                reeval=reeval_now,
                                build_cache=(want_cache and not is_decode),
                                mamba_inference_params=mamba_inf,
                                real_pos=real_pos,
                                selected_per_seq=sel_ps,
                                advanced_per_seq=adv_ps,
                                reeval_mask=reeval_rows)
                            new_conv_seq = None
                        if is_decode or want_cache:
                            conv_caches[i] = new_conv
                            if track_seq_cache:
                                conv_seq_caches[i] = new_conv_seq
                            if new_attn_cache is not None:
                                thought_caches[i] = new_attn_cache
                        return h_out
                return layer_fn

            layer_fn = make_layer_fn(i, block, is_global, is_last_global, sel_adv, sel_adv_seq, router_outputs, router_per_seq)

            if self.use_checkpoint and self.training and torch.is_grad_enabled() and not is_decode:
                def wrapped(h, _layer_fn=layer_fn):
                    return checkpoint(_layer_fn, h, use_reentrant=True)
            else:
                wrapped = layer_fn

            streams = self.hcs[i](streams, wrapped)

        w = F.softmax(self.final_mix, dim=0)
        x = sum(w[j] * streams[j] for j in range(self.num_streams))
        x = self.norm(x)

        if mamba_inf is not None:
            mamba_inf.seqlen_offset += T_step

        new_cache = None
        if want_cache:
            new_seq_len = (past_seq_len + T_step) if is_decode else T_step
            last_sel = cache.get("last_sel", None) if is_decode else None
            last_adv = cache.get("last_adv", None) if is_decode else None
            for j in range(self.num_blocks - 1, -1, -1):
                if router_outputs[j] is not None:
                    last_sel, last_adv = router_outputs[j]
                    break
            new_cache = {
                "attn_kvs": attn_kvs,
                "conv_caches": conv_caches,
                "thought_caches": thought_caches,
                "mamba_inference_params": mamba_inf,
                "router_routes": router_outputs,
                "last_sel": last_sel,
                "last_adv": last_adv,
                "seq_len": new_seq_len,
                "row_seq_lens": row_seq_lens_after,
                "attn_kv_pos": full_kv_pos,
                "active_seq_ids": active_seq_ids,
                "attn_seq_idxs": attn_seq_idxs,
                "conv_seq_caches": conv_seq_caches,
            }
        return x, new_cache


class CouncilModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CouncilBackbone(
            d_model=config.d_model_backbone,
            d_model_thought=config.d_model_thought,
            num_blocks=config.num_blocks,
            num_streams=config.num_streams,
            vocab_size=config.vocab_size,
            use_checkpoint=getattr(config, "backbone_checkpoint", False),
            window_size=getattr(config, "sliding_window", 4096),
            attention_checkpoint=getattr(config, "attention_checkpoint", None),
            config=config)
        self.backbone.config_mamba_headdim = config.mamba_headdim
        self.final_norm = nn.LayerNorm(config.d_model_backbone)
        self.lm_head = nn.Linear(config.d_model_backbone, config.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.backbone.embedding.weight, mean=0, std=0.02)

    def forward(self, input_ids, labels=None, num_recursion=None, cache=None,
                return_cache=False, force_full_recursion=False, cu_seqlens=None,
                attention_mask=None, return_loss_info=False):
        x, new_cache = self.backbone(input_ids, cache=cache, return_cache=return_cache,
                                       cu_seqlens=cu_seqlens, attention_mask=attention_mask)
        x_out = self.final_norm(x)
        logits = self.lm_head(x_out)
        if labels is None:
            if return_cache:
                return logits, new_cache
            return logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:].clone()
        if cu_seqlens is not None:
            if cu_seqlens.dim() == 1:
                cu_seqlens = cu_seqlens.unsqueeze(0)
            B_labels = labels.size(0)
            B_cu = cu_seqlens.size(0)
            if B_cu == 1 and B_labels > 1:
                cu_eff = cu_seqlens.expand(B_labels, -1)
            elif B_cu == B_labels:
                cu_eff = cu_seqlens
            else:
                raise ValueError(f"cu_seqlens batch size {B_cu} does not match labels batch size {B_labels}")
            for b in range(B_labels):
                row = cu_eff[b].to(torch.long)
                valid_vals = row[row > 0]
                if valid_vals.numel() > 0:
                    max_end = int(valid_vals.max().item())
                    if max_end < labels.size(1):
                        shift_labels[b, max(max_end - 1, 0):] = -100
                elif (row >= 0).any() and row.numel() >= 2:
                    shift_labels[b, :] = -100
                pos = row[1:] - 1
                valid = (pos >= 0) & (pos < shift_labels.size(1))
                pos = pos[valid]
                shift_labels[b, pos] = -100
        if attention_mask is not None and attention_mask.size(1) == labels.size(1):
            shift_am_src = attention_mask[..., :-1]
            shift_am_tgt = attention_mask[..., 1:]
            shift_labels[shift_am_src == 0] = -100
            shift_labels[shift_am_tgt == 0] = -100
        B_, T_, V_ = shift_logits.shape
        chunk_size = 1024
        total_loss = torch.zeros((), device=shift_logits.device, dtype=torch.float32)
        total_valid = torch.zeros((), device=shift_logits.device, dtype=torch.float32)
        flat_logits = shift_logits.reshape(-1, V_)
        flat_labels = shift_labels.reshape(-1)
        for i in range(0, B_ * T_, chunk_size):
            cl = flat_logits[i:i + chunk_size].float()
            cla = flat_labels[i:i + chunk_size]
            chunk_loss = F.cross_entropy(cl, cla, ignore_index=-100, reduction='sum')
            total_loss = total_loss + chunk_loss
            total_valid = total_valid + (cla != -100).sum()
        loss = total_loss / total_valid.clamp(min=1)
        if return_loss_info:
            if return_cache:
                return logits, loss, total_valid, new_cache
            return logits, loss, total_valid
        if return_cache:
            return logits, loss, new_cache
        return logits, loss

    @torch.no_grad()
    def probe_recursion(self, input_ids, labels, max_rec=None, delta_loss_threshold=0.05):
        return None

    @staticmethod
    def _apply_repetition_penalty(logits, generated_ids, penalty, attention_mask=None):
        if penalty is None or penalty <= 1.0:
            return logits
        if attention_mask is None:
            score = torch.gather(logits, 1, generated_ids)
            score = torch.where(score < 0, score * penalty, score / penalty)
            return logits.scatter(1, generated_ids, score)
        logits = logits.clone()
        B = generated_ids.size(0)
        am = attention_mask.bool()
        for b in range(B):
            ids_b = generated_ids[b][am[b]]
            if ids_b.numel() == 0:
                continue
            ids_b = torch.unique(ids_b)
            score = logits[b, ids_b]
            score = torch.where(score < 0, score * penalty, score / penalty)
            logits[b, ids_b] = score
        return logits

    @staticmethod
    def _apply_no_repeat_ngram(logits, generated_ids, n, attention_mask=None):
        if n is None or n <= 0:
            return logits
        B = generated_ids.size(0)
        logits = logits.clone()
        am = attention_mask.bool() if attention_mask is not None else None
        for b in range(B):
            if am is not None:
                seq = generated_ids[b][am[b]].tolist()
            else:
                seq = generated_ids[b].tolist()
            if len(seq) < n:
                continue
            prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
            banned = set()
            for i in range(len(seq) - n + 1):
                ng = tuple(seq[i:i + n])
                if ng[:n - 1] == prefix:
                    banned.add(ng[-1])
            if banned:
                idx = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
                logits[b, idx] = float("-inf")
        return logits

    def _sample_next(self, nxt_logits, temperature):
        if temperature is None or temperature <= 0:
            return nxt_logits.argmax(dim=-1, keepdim=True)
        nxt_logits = nxt_logits.float() / max(temperature, 1e-6)
        probs = F.softmax(nxt_logits, dim=-1)
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        row_sum = probs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        probs = torch.where(row_sum > 0, probs / row_sum.clamp(min=1e-12), uniform)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=512,
                 temperature=0.8, eos_token_id=None, repetition_penalty=1.15,
                 no_repeat_ngram_size=0, cu_seqlens=None):
        self.eval()
        B = input_ids.size(0)
        T_p = input_ids.size(1)
        device = input_ids.device
        if T_p == 0:
            raise ValueError("input_ids must have non-zero sequence length")
        if attention_mask is None:
            attention_mask = torch.ones(B, T_p, device=device, dtype=torch.long)
        else:
            attention_mask = attention_mask.long()
        am_bool = attention_mask[:, :T_p].bool()
        if cu_seqlens is not None:
            seq_idx_p = self.backbone._build_seq_idx_from_cu_seqlens(
                cu_seqlens, T_p, attention_mask, device, target_batch=B)
            active_seq_p = self.backbone._extract_active_seq_ids(seq_idx_p, T_p)
            token_mask_p = am_bool & (seq_idx_p == active_seq_p.view(B, 1)) & (active_seq_p.view(B, 1) >= 0)
        else:
            token_mask_p = am_bool
        if not token_mask_p.any(dim=1).all().item():
            raise ValueError("Each prompt in generate() must contain at least one non-padding token")
        if max_new_tokens <= 0:
            return input_ids

        k = self.backbone.engram.max_ngram - 1
        cfg_best_N = getattr(self.config, "best_N", 0)
        n_rec = cfg_best_N if cfg_best_N > 0 else self.config.rl_generate_recursion
        force_full = not getattr(self.config, "adaptive_inference", True)

        logits, cache = self.forward(input_ids, attention_mask=attention_mask,
                                       num_recursion=n_rec, force_full_recursion=force_full,
                                       cu_seqlens=cu_seqlens, return_cache=True)

        pos_grid = torch.arange(T_p, device=device).unsqueeze(0).expand(B, T_p)
        masked_pos = torch.where(token_mask_p, pos_grid, torch.full_like(pos_grid, -1))
        last_pos = masked_pos.max(dim=1).values
        if k > 0:
            last_tokens = torch.full((B, k), -1, device=device, dtype=input_ids.dtype)
            lengths = token_mask_p.long().sum(dim=1)
            rank = token_mask_p.long().cumsum(dim=1)
            valid_src = token_mask_p & (rank > (lengths.unsqueeze(1) - k))
            if valid_src.any():
                b_idx, t_idx = valid_src.nonzero(as_tuple=True)
                dst_idx = rank[b_idx, t_idx] - lengths[b_idx] + (k - 1)
                last_tokens[b_idx, dst_idx] = input_ids[b_idx, t_idx]
        else:
            last_tokens = input_ids.new_zeros(B, 0)
        out = input_ids
        penalty_mask = token_mask_p.long()
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        nxt_logits = logits[torch.arange(B, device=device), last_pos, :]
        nxt_logits = self._apply_repetition_penalty(nxt_logits, out, repetition_penalty, attention_mask=penalty_mask)
        nxt_logits = self._apply_no_repeat_ngram(nxt_logits, out, no_repeat_ngram_size, attention_mask=penalty_mask)
        nxt = self._sample_next(nxt_logits, temperature)
        if eos_token_id is not None:
            nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_token_id), nxt)
        out = torch.cat([out, nxt], dim=-1)
        if eos_token_id is not None:
            finished = finished | (nxt.squeeze(-1) == eos_token_id)
            if finished.all().item():
                return out
        for _ in range(max_new_tokens - 1):
            cache["last_tokens"] = last_tokens
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, device=device, dtype=attention_mask.dtype)],
                dim=1)
            penalty_mask = torch.cat(
                [penalty_mask, torch.ones(B, 1, device=device, dtype=penalty_mask.dtype)],
                dim=1)
            logits, cache = self.forward(nxt, attention_mask=attention_mask,
                                           num_recursion=n_rec, cache=cache,
                                           force_full_recursion=force_full, return_cache=True)
            if k > 0:
                last_tokens = torch.cat([last_tokens, nxt], dim=1)[:, -k:]
            else:
                last_tokens = input_ids.new_zeros(B, 0)
            nxt_logits = logits[:, -1, :]
            nxt_logits = self._apply_repetition_penalty(nxt_logits, out, repetition_penalty, attention_mask=penalty_mask)
            nxt_logits = self._apply_no_repeat_ngram(nxt_logits, out, no_repeat_ngram_size, attention_mask=penalty_mask)
            nxt = self._sample_next(nxt_logits, temperature)
            if eos_token_id is not None:
                nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_token_id), nxt)
            out = torch.cat([out, nxt], dim=-1)
            if eos_token_id is not None:
                finished = finished | (nxt.squeeze(-1) == eos_token_id)
                if finished.all().item():
                    break
        return out