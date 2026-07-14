"""PE-AV: a from-scratch, educational reimplementation of Meta AI's Perception
Encoder Audiovisual.

Quickstart
----------
>>> import torch
>>> from pe_av import PEAV, PEAVConfig
>>> model = PEAV(PEAVConfig.preset("small"))
>>> frames = torch.randn(2, 8, 3, 64, 64)
>>> spec = torch.randn(2, 64, 128)
>>> out = model(frames=frames, spectrogram=spec)
>>> out.av_embed.shape
torch.Size([2, 192])
"""

from .config import (
    AudioEncoderConfig,
    FrameEncoderConfig,
    FusionEncoderConfig,
    PEAVConfig,
    TextEncoderConfig,
    VideoEncoderConfig,
)
from .losses import MultiPairContrastiveLoss, frame_level_contrastive, info_nce
from .modeling import PEAV, PEAVOutput
from .retrieval import EmbeddingIndex

__version__ = "0.1.0"

__all__ = [
    "PEAV",
    "PEAVOutput",
    "PEAVConfig",
    "FrameEncoderConfig",
    "VideoEncoderConfig",
    "AudioEncoderConfig",
    "TextEncoderConfig",
    "FusionEncoderConfig",
    "MultiPairContrastiveLoss",
    "frame_level_contrastive",
    "info_nce",
    "EmbeddingIndex",
    "__version__",
]
