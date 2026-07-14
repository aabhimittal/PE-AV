"""Frame encoder: a Vision Transformer that turns a single RGB frame into tokens.

This is the image tower.  A ``[CLS]`` token summarises the frame; the per-patch
tokens are returned as well so the video tower can attend over spatial detail if
desired.  In the real PE-AV release this corresponds to the ``PE-Core`` image
backbone; here it is a compact but faithful ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FrameEncoderConfig
from .common import Transformer


class FrameEncoder(nn.Module):
    def __init__(self, cfg: FrameEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.grid = cfg.image_size // cfg.patch_size
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.width, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.width))
        self.dropout = nn.Dropout(cfg.dropout)
        self.transformer = Transformer(cfg.width, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.width)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.cfg.width

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``frames`` of shape ``(B, C, H, W)``.

        Returns ``(cls, patch_tokens)`` where ``cls`` is ``(B, width)`` and
        ``patch_tokens`` is ``(B, num_patches, width)``.
        """
        b = frames.shape[0]
        x = self.patch_embed(frames)  # (B, width, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, width)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]
