"""Text encoder: a bidirectional transformer over token ids.

Captions describing speech / music / sound-effects are embedded here.  A learned
``[CLS]`` token (prepended) yields the pooled text embedding used for
audio-text and video-text contrastive alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import TextEncoderConfig
from .common import Transformer


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.width)
        # +1 position for the prepended CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_length + 1, cfg.width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.dropout = nn.Dropout(cfg.dropout)
        self.transformer = Transformer(cfg.width, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.width)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.cfg.width

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``token_ids`` of shape ``(B, L)``.

        ``attention_mask`` is ``(B, L)`` with 1 for real tokens, 0 for padding
        (HuggingFace convention).  Returns ``(cls, token_features)``.
        """
        b, length = token_ids.shape
        if length > self.cfg.max_length:
            raise ValueError(f"Sequence length {length} exceeds max_length {self.cfg.max_length}.")
        x = self.token_embed(token_ids)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, width)
        x = x + self.pos_embed[:, : length + 1]
        x = self.dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            pad = attention_mask == 0  # True where padding
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=pad.device)
            key_padding_mask = torch.cat([cls_mask, pad], dim=1)

        x = self.transformer(x, self_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]
