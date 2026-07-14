"""Audio encoder: an Audio Spectrogram Transformer (AST-style) tower.

Input is a log-mel spectrogram ``(B, n_mels, T)``.  It is patchified in the
time-frequency plane, flattened into a token sequence, and processed by a
transformer with a ``[CLS]`` summary token.  A padding mask over time lets us
batch variable-length clips.  This is PE-AV's audio branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AudioEncoderConfig
from .common import Transformer


class AudioEncoder(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_mels % cfg.patch_freq != 0:
            raise ValueError("n_mels must be divisible by patch_freq.")
        self.freq_patches = cfg.n_mels // cfg.patch_freq
        self.patch_embed = nn.Conv2d(
            1, cfg.width,
            kernel_size=(cfg.patch_freq, cfg.patch_time),
            stride=(cfg.patch_freq, cfg.patch_time),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.width))
        max_time_patches = cfg.max_frames // cfg.patch_time
        max_tokens = self.freq_patches * max_time_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, cfg.width))
        self.dropout = nn.Dropout(cfg.dropout)
        self.transformer = Transformer(cfg.width, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.width)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.cfg.width

    def forward(
        self, spectrogram: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a log-mel ``spectrogram`` of shape ``(B, n_mels, T)``.

        ``time_mask`` is ``(B, T)`` with True at padded time steps.  Returns
        ``(cls, patch_tokens)``.
        """
        b, n_mels, t = spectrogram.shape
        pt = self.cfg.patch_time
        # Right-pad time to a multiple of the patch size so patching is exact.
        if t % pt != 0:
            pad = pt - (t % pt)
            spectrogram = nn.functional.pad(spectrogram, (0, pad))
            if time_mask is not None:
                time_mask = nn.functional.pad(time_mask, (0, pad), value=True)
            t = spectrogram.shape[-1]

        x = spectrogram.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.patch_embed(x)  # (B, width, freq_patches, time_patches)
        _, _, fp, tp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, fp*tp, width)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.dropout(x)

        key_padding_mask = None
        if time_mask is not None:
            # Collapse the per-sample time mask to per-time-patch, then broadcast
            # across frequency patches to match the flattened token order.
            tm = time_mask[:, :t].view(b, t // pt, pt).all(dim=-1)  # (B, tp)
            patch_mask = tm.unsqueeze(1).expand(b, fp, tp).reshape(b, fp * tp)
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=patch_mask.device)
            key_padding_mask = torch.cat([cls_mask, patch_mask], dim=1)

        x = self.transformer(x, self_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]
