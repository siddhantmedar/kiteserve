"""Qwen3 forward pass, with attention that reads and writes the paged KV cache.

Shapes (Qwen3-0.6B):  N = number of tokens in the batch (flat, no batch dim)
    hidden = 1024, heads = 16, kv_heads = 8, head_dim = 128, layers = 28, vocab = 151936
    Note: heads * head_dim = 2048, which is BIGGER than hidden (1024). That's fine:
    q_proj maps 1024 -> 2048 and o_proj maps 2048 -> 1024.

Attribute names MUST match HuggingFace (the loader copies weights by name):
    model.embed_tokens, model.layers[i], model.norm, lm_head
    model.layers[i].input_layernorm / post_attention_layernorm
    model.layers[i].self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}
    model.layers[i].mlp.{gate_proj, up_proj, down_proj}
All Linear layers have bias=False. RoPE angles are computed on the fly, so there are no buffers
for the loader to skip. Op order follows transformers/models/qwen3/modeling_qwen3.py.

attn_meta / kv_cache: both None = plain forward with no cache (recompute everything).
    Otherwise Qwen3Attention.forward does prefill (write + causal) or decode (write + gather + length mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiteserve.config import ModelConfig


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight.   One learned weight per feature.

    The mean and sqrt run in float32 even when x is bf16; the result is cast back.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        den = torch.sqrt(torch.mean(x.to(torch.float32)**2,dim=-1,keepdim=True) + self.eps)
        return ((x.to(torch.float32)/ den).to(x.dtype) * self.weight)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotate q or k by their positions.  x: [N, num_heads, head_dim], positions: [N].

    Qwen uses the "rotate_half" layout: pair dimension i with dimension i + head_dim/2.
    Frequencies: inv_freq[j] = 1 / theta ** (2j / head_dim)  for j in 0 .. head_dim/2 - 1.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2

    j = torch.arange(half,device=x.device,dtype=torch.float32)
    inv_freq = 1/theta ** (2*j / head_dim)

    angles = positions.float()[:,None] * inv_freq[None,:] # (N, half)

    cos, sin = angles.cos()[:,None,:], angles.sin()[:,None,:]

    x1 = x[...,:half].float()
    x2 = x[...,half:].float()

    return torch.cat([
        cos*x1 - sin*x2,
        sin*x1 + cos*x2
    ],dim=-1).to(x.dtype) # [N, num_heads, head_dim]

class Qwen3Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx  # which layer of the KV cache this attention writes/reads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, attn_meta=None, kv_cache=None) -> torch.Tensor:
        """x: [N, hidden] -> [N, hidden].
        N: number of tokens we are processing in this forward pass
        Steps: project -> split into heads -> q_norm/k_norm (per head!) -> RoPE
               -> attention (16 query heads share 8 kv heads: GQA) -> merge heads -> o_proj
        Attention by case:
            no cache:  causal over the fresh k, v
            prefill:   write k, v to the cache, then causal over the fresh k, v
            decode:    write, gather history [B, L, ...], mask positions >= context_lens (never is_causal)
        SDPA wants [batch, heads, seq, head_dim]; we have [N, heads, head_dim].
        """
        q = self.q_proj(x) # [N, hidden] @ [hidden, num_heads * head_dim] -> [N, num_heads * head_dim]
        k = self.k_proj(x) # [N, hidden] @ [hidden, num_kv_heads * head_dim] -> [N, num_kv_heads * head_dim]
        v = self.v_proj(x) # [N, hidden] @ [hidden, num_kv_heads * head_dim] -> [N, num_kv_heads * head_dim]

        q = q.contiguous().view(q.size(0), self.cfg.num_heads, self.cfg.head_dim) # [N, num_heads * head_dim] -> [N, num_heads, head_dim]
        k = k.contiguous().view(k.size(0), self.cfg.num_kv_heads, self.cfg.head_dim) # [N, num_kv_heads * head_dim] -> [N, num_kv_heads, head_dim]
        v = v.contiguous().view(v.size(0), self.cfg.num_kv_heads, self.cfg.head_dim) # [N, num_kv_heads * head_dim] -> [N, num_kv_heads, head_dim]

        q = self.q_norm(q) # [N, num_heads, head_dim] -> same (per-head RMSNorm)
        k = self.k_norm(k) # [N, num_kv_heads, head_dim] -> same (per-head RMSNorm)

        q = apply_rope(q, positions, self.cfg.rope_theta) # [N, num_heads, head_dim] -> same, with RoPE applied
        k = apply_rope(k, positions, self.cfg.rope_theta) # [N, num_kv_heads, head_dim] -> same, with RoPE applied

        if attn_meta is not None:
            kv_cache.write(self.layer_idx, k, v, attn_meta.slot_mapping)

        if attn_meta is None or attn_meta.is_prefill:
            q = q.transpose(0,1).unsqueeze(0) # [N, num_heads, head_dim] -> [1, num_heads, N, head_dim] for SDPA compatibility
            k = k.transpose(0,1).unsqueeze(0) # [N, num_kv_heads, head_dim] -> [1, num_kv_heads, N, head_dim] for SDPA compatibility
            v = v.transpose(0,1).unsqueeze(0) # [N, num_kv_heads, head_dim] -> [1, num_kv_heads, N, head_dim] for SDPA compatibility

            attn_out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=True) # [1, num_heads, N, head_dim] x [1, num_kv_heads, N, head_dim] -> [1, num_heads, N, head_dim]

            attn_out = attn_out.squeeze(0).transpose(0,1) # [1, num_heads, N, head_dim] -> [N, num_heads, head_dim]
            attn_out = attn_out.contiguous().view(x.size(0), -1) # [N, num_heads, head_dim] -> [N, num_heads * head_dim]

            return self.o_proj(attn_out) # [N, num_heads * head_dim] -> [N, hidden] via output projection

        # Decode: gather each sequence's history through its block table (L = M * block_size, padded)
        k_cache,v_cache = kv_cache.gather(self.layer_idx, attn_meta.block_tables, attn_meta.context_lens) # [B, L, kv_heads, head_dim] for K and V respectively
        q = q.unsqueeze(2) # [N, num_heads, head_dim] -> [N, num_heads, 1, head_dim] for SDPA compatibility
        # SDPA expects [batch, heads, seq, head_dim] and attends over the second-to-last dim (seq).
        # The cache gather returns [B, L, kv_heads, head_dim] (seq before heads), so swap dims 1 and 2
        # so that heads are a batch-like dim and L is the sequence axis that q attends over.
        k_cache = k_cache.transpose(1,2) # [B, L, kv_heads, head_dim] -> [B, kv_heads, L, head_dim] for SDPA compatibility
        v_cache = v_cache.transpose(1,2) # [B, L, kv_heads, head_dim] -> [B, kv_heads, L, head_dim] for SDPA compatibility
        context_lens = attn_meta.context_lens[:,None] # [B] -> [B, 1] for broadcasting with SDPA attention mask
        key_positions = torch.arange(k_cache.size(2), device=k_cache.device) # [L] sequence positions for SDPA attention mask
        attn_mask = context_lens > key_positions # [B, 1] > [L] -> [B, L] boolean mask for SDPA attention
        attn_mask = attn_mask[:,None,None,:] # [B, L] -> [B, 1, 1, L] for SDPA attention mask broadcasting
        # SDPA expects attn_mask of shape [batch, heads, query_seq, key_seq], where query_seq=1 for decoding.
        # The mask will be broadcasted correctly over the query dimension.
        attn_out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=attn_mask, is_causal=False, enable_gqa=True) # [B, num_heads, 1, head_dim]
        attn_out = attn_out.squeeze(2) # [B, num_heads, 1, head_dim] -> [B, num_heads, head_dim] for merging heads
        attn_out = attn_out.reshape(attn_out.size(0), -1) # [B, num_heads, head_dim] -> [B, num_heads * head_dim] for merging heads
        return self.o_proj(attn_out) # [B, num_heads * head_dim] -> [B, hidden_size] after output projection


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size,bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size,bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size,bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU:  down( silu(gate(x)) * up(x) )"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)
        self.layer_idx = layer_idx
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor, attn_meta=None, kv_cache=None) -> torch.Tensor:
        """Pre-norm residual block:  x = x + attn(norm(x));  x = x + mlp(norm(x)).  Return a tensor."""
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, attn_meta=attn_meta, kv_cache=kv_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size,cfg.rms_norm_eps)

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,  # [N] token ids
        positions: torch.Tensor,  # [N] position of each token in its sequence
        attn_meta=None,  # kiteserve.kv_cache.AttnMeta | None
        kv_cache=None,  # kiteserve.kv_cache.KVCache | None
    ) -> torch.Tensor:
        """Returns logits [N, vocab]:  embed -> 28 decoder layers -> final norm -> lm_head."""
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, positions, attn_meta=attn_meta, kv_cache=kv_cache)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return logits # [N, vocab]