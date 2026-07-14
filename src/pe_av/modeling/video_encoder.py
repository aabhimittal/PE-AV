"""Video encoder: aggregates a sequence of per-frame embeddings into one clip vector.

The frame tower is applied to each frame independently; this tower adds temporal
position embeddings and a learnable ``[VID]`` token, then runs a small temporal
transformer.  This mirrors PE-AV's "frame encoder + video encoder" split, which
lets the expensive per-frame ViT be shared while temporal reasoning stays cheap.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import VideoEncoderConfig
from .common import Transformer, sinusoidal_position_embedding


class VideoEncoder(nn.Module):
    def __init__(self, cfg: VideoEncoderConfig, frame_dim: int):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(frame_dim, cfg.width) if frame_dim != cfg.width else nn.Identity()
        self.vid_token = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.register_buffer(
            "temporal_pos",
            sinusoidal_position_embedding(cfg.max_frames + 1, cfg.width),
            persistent=False,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.transformer = Transformer(cfg.width, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.width)
        nn.init.trunc_normal_(self.vid_token, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.cfg.width

    def forward(
        self, frame_embeds: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate ``frame_embeds`` of shape ``(B, T, frame_dim)``.

        ``frame_mask`` is ``(B, T)`` with True where a frame is *padding*.
        Returns ``(clip_embed, frame_tokens)`` where ``clip_embed`` is
        ``(B, width)`` and ``frame_tokens`` is ``(B, T, width)`` (temporally
        contextualised per-frame features, used by PE-A-Frame).
        """
        b, t, _ = frame_embeds.shape
        x = self.in_proj(frame_embeds)
        vid = self.vid_token.expand(b, -1, -1)
        x = torch.cat([vid, x], dim=1)  # (B, T+1, width)
        x = x + self.temporal_pos[: t + 1].unsqueeze(0)
        x = self.dropout(x)

        key_padding_mask = None
        if frame_mask is not None:
            vid_mask = torch.zeros(b, 1, dtype=torch.bool, device=frame_mask.device)
            key_padding_mask = torch.cat([vid_mask, frame_mask], dim=1)

        x = self.transformer(x, self_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]
