"""Reusable transformer building blocks shared by every PE-AV tower.

Everything here is deliberately small and dependency-free (plain ``torch.nn``)
so the architecture is easy to read end-to-end.  The blocks use pre-norm
residual connections, which is the standard recipe in modern ViT / CLIP style
encoders.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Position-wise feed-forward network (GELU)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting optional cross-attention and key padding masks."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = query if key is None else key
        value = key if value is None else value

        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, h, nq, nk)
        if key_padding_mask is not None:
            # key_padding_mask: (b, nk) with True where the key should be ignored.
            mask = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (b, h, nq, head_dim)
        b, _, nq, _ = out.shape
        out = out.transpose(1, 2).reshape(b, nq, self.heads * self.head_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block; set ``cross_attention`` for encoder-decoder use."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 cross_attention: bool = False):
        super().__init__()
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        if cross_attention:
            self.norm_ctx = nn.LayerNorm(dim)
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=self_key_padding_mask)
        if self.cross_attention and context is not None:
            q = self.norm_cross(x)
            kv = self.norm_ctx(context)
            x = x + self.cross_attn(q, kv, kv, key_padding_mask=context_key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """A stack of :class:`TransformerBlock`."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, cross_attention: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout, cross_attention) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                self_key_padding_mask: torch.Tensor | None = None,
                context_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, context, self_key_padding_mask, context_key_padding_mask)
        return x


def sinusoidal_position_embedding(length: int, dim: int, device=None) -> torch.Tensor:
    """Classic transformer sinusoidal positional embedding, shape ``(length, dim)``."""
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalise embeddings so dot products become cosine similarities."""
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)
