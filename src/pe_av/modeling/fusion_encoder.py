"""Audio-Video fusion encoder.

PE-AV produces not just separate audio and video embeddings but a *joint*
audiovisual embedding.  This tower cross-attends the two streams and pools a
learnable ``[FUSE]`` token into a single vector living in the same shared space
as the unimodal embeddings.

Novel element — **correspondence gating**.  Natural audio and video are only
sometimes in correspondence (a dubbed clip, off-screen narration, background
music).  Before fusing, we compute a per-sample audio-video agreement score
``g = sigmoid(w * cos(a, v) + b)`` and use it to gate how much cross-modal
information flows into the fused token.  When audio and video disagree, the gate
closes and the fused embedding falls back toward the visual stream, which
empirically stabilises retrieval on weakly-aligned web data.  See
``docs/CONCEPTS.md`` for the derivation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FusionEncoderConfig
from .common import Transformer, l2_normalize


class FusionEncoder(nn.Module):
    def __init__(self, cfg: FusionEncoderConfig, audio_dim: int, video_dim: int):
        super().__init__()
        self.cfg = cfg
        self.audio_proj = nn.Linear(audio_dim, cfg.width)
        self.video_proj = nn.Linear(video_dim, cfg.width)
        self.fuse_token = nn.Parameter(torch.zeros(1, 1, cfg.width))
        # Modality-type embeddings distinguish audio vs. video tokens in the joint set.
        self.type_embed = nn.Parameter(torch.zeros(3, cfg.width))  # 0=fuse,1=audio,2=video
        self.transformer = Transformer(
            cfg.width, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout
        )
        self.norm = nn.LayerNorm(cfg.width)

        if cfg.use_correspondence_gate:
            self.gate_scale = nn.Parameter(torch.tensor(1.0))
            self.gate_bias = nn.Parameter(torch.tensor(0.0))

        nn.init.trunc_normal_(self.fuse_token, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.cfg.width

    def correspondence(self, audio_embed: torch.Tensor, video_embed: torch.Tensor) -> torch.Tensor:
        """Return the per-sample audio-video correspondence gate in ``(0, 1)``.

        ``audio_embed`` / ``video_embed`` are the *shared-space* unimodal
        embeddings ``(B, D)``.  Returns ``(B, 1)``.
        """
        cos = (l2_normalize(audio_embed) * l2_normalize(video_embed)).sum(-1, keepdim=True)
        return torch.sigmoid(self.gate_scale * cos + self.gate_bias)

    def forward(
        self,
        audio_cls: torch.Tensor,
        video_cls: torch.Tensor,
        audio_tokens: torch.Tensor | None = None,
        video_tokens: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse audio and video into one vector of shape ``(B, width)``.

        ``*_cls`` are the tower summary embeddings; ``*_tokens`` are the optional
        full token sequences (frame tokens / audio patch tokens) for finer fusion.
        ``gate`` optionally overrides the internal correspondence gate.
        """
        b = audio_cls.shape[0]

        audio_seq = audio_cls.unsqueeze(1) if audio_tokens is None else torch.cat(
            [audio_cls.unsqueeze(1), audio_tokens], dim=1
        )
        video_seq = video_cls.unsqueeze(1) if video_tokens is None else torch.cat(
            [video_cls.unsqueeze(1), video_tokens], dim=1
        )

        a = self.audio_proj(audio_seq) + self.type_embed[1]
        v = self.video_proj(video_seq) + self.type_embed[2]

        if self.cfg.use_correspondence_gate:
            if gate is None:
                gate = self.correspondence(audio_cls, video_cls)
            # Gate the audio contribution: closing it makes fusion visual-dominant.
            a = a * gate.unsqueeze(1)

        fuse = self.fuse_token.expand(b, -1, -1) + self.type_embed[0]
        x = torch.cat([fuse, a, v], dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]
