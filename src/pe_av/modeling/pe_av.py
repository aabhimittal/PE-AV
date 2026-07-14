"""The full PE-AV model.

Five towers (frame, video, audio, fusion, text) each end in a linear projection
into a **shared** ``embed_dim`` space where all modalities are L2-normalised and
compared with a single learnable temperature.  This is what makes any-to-any
retrieval (audio->text, video->audio, text->audiovisual, ...) a single dot
product.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import PEAVConfig
from .audio_encoder import AudioEncoder
from .common import l2_normalize
from .frame_encoder import FrameEncoder
from .fusion_encoder import FusionEncoder
from .text_encoder import TextEncoder
from .video_encoder import VideoEncoder


@dataclass
class PEAVOutput:
    """Container for all embeddings produced in a forward pass.

    Every ``*_embed`` is L2-normalised and lives in the shared space, so any two
    can be compared directly.  Fields are ``None`` when the corresponding input
    was not provided.  ``frame_embeds`` holds per-frame shared-space vectors used
    by PE-A-Frame for temporal localisation.
    """

    audio_embed: torch.Tensor | None = None
    video_embed: torch.Tensor | None = None
    text_embed: torch.Tensor | None = None
    av_embed: torch.Tensor | None = None
    frame_embeds: torch.Tensor | None = None
    logit_scale: torch.Tensor | None = None


class PEAV(nn.Module):
    def __init__(self, cfg: PEAVConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = FrameEncoder(cfg.frame)
        self.video_encoder = VideoEncoder(cfg.video, frame_dim=self.frame_encoder.output_dim)
        self.audio_encoder = AudioEncoder(cfg.audio)
        self.text_encoder = TextEncoder(cfg.text)
        self.fusion_encoder = FusionEncoder(
            cfg.fusion,
            audio_dim=self.audio_encoder.output_dim,
            video_dim=self.video_encoder.output_dim,
        )

        d = cfg.embed_dim
        self.video_head = nn.Linear(self.video_encoder.output_dim, d, bias=False)
        self.frame_head = nn.Linear(self.frame_encoder.output_dim, d, bias=False)
        self.audio_head = nn.Linear(self.audio_encoder.output_dim, d, bias=False)
        self.text_head = nn.Linear(self.text_encoder.output_dim, d, bias=False)
        self.av_head = nn.Linear(self.fusion_encoder.output_dim, d, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale_init))

    # ---- individual modality encoders (all return shared-space, L2-normed) ----
    def encode_video(self, frames: torch.Tensor, frame_mask: torch.Tensor | None = None):
        """``frames``: ``(B, T, C, H, W)``. Returns ``(clip_embed, frame_embeds, raw)``."""
        b, t = frames.shape[:2]
        flat = frames.flatten(0, 1)  # (B*T, C, H, W)
        frame_cls, _ = self.frame_encoder(flat)
        frame_cls = frame_cls.view(b, t, -1)
        clip_raw, frame_tokens = self.video_encoder(frame_cls, frame_mask)
        clip_embed = l2_normalize(self.video_head(clip_raw))
        frame_embeds = l2_normalize(self.frame_head(frame_tokens))
        return clip_embed, frame_embeds, (clip_raw, frame_tokens)

    def encode_audio(self, spectrogram: torch.Tensor, time_mask: torch.Tensor | None = None):
        """``spectrogram``: ``(B, n_mels, T)``. Returns ``(audio_embed, raw_cls, tokens)``."""
        audio_cls, audio_tokens = self.audio_encoder(spectrogram, time_mask)
        audio_embed = l2_normalize(self.audio_head(audio_cls))
        return audio_embed, audio_cls, audio_tokens

    def encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        text_cls, _ = self.text_encoder(token_ids, attention_mask)
        return l2_normalize(self.text_head(text_cls))

    def fuse(self, audio_cls, video_cls, audio_tokens=None, video_tokens=None,
             audio_embed=None, video_embed=None):
        gate = None
        if (self.cfg.fusion.use_correspondence_gate and audio_embed is not None
                and video_embed is not None):
            gate = self.fusion_encoder.correspondence(audio_embed, video_embed)
        av_raw = self.fusion_encoder(audio_cls, video_cls, audio_tokens, video_tokens, gate=gate)
        return l2_normalize(self.av_head(av_raw))

    def clamp_logit_scale(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.cfg.logit_scale_max)

    def forward(
        self,
        frames: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        spectrogram: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        compute_fusion: bool = True,
    ) -> PEAVOutput:
        out = PEAVOutput(logit_scale=self.logit_scale.exp())

        video_cls_raw = frame_tokens = None
        if frames is not None:
            out.video_embed, out.frame_embeds, (video_cls_raw, frame_tokens) = self.encode_video(
                frames, frame_mask
            )

        audio_cls_raw = audio_tokens = None
        if spectrogram is not None:
            out.audio_embed, audio_cls_raw, audio_tokens = self.encode_audio(spectrogram, time_mask)

        if token_ids is not None:
            out.text_embed = self.encode_text(token_ids, attention_mask)

        if compute_fusion and frames is not None and spectrogram is not None:
            out.av_embed = self.fuse(
                audio_cls_raw, video_cls_raw, audio_tokens, frame_tokens,
                audio_embed=out.audio_embed, video_embed=out.video_embed,
            )
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_preset(cls, name: str) -> PEAV:
        return cls(PEAVConfig.preset(name))
