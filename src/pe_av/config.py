"""Configuration dataclasses for PE-AV.

The whole model is described by a single :class:`PEAVConfig`, which nests one
config per encoder tower plus shared projection / training options.  Configs can
be created in code, loaded from YAML, or picked from the named presets in
``pe_av.config.PRESETS`` (``small`` / ``base`` / ``large``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FrameEncoderConfig:
    """Per-frame Vision Transformer (image tower)."""

    image_size: int = 64
    patch_size: int = 16
    in_channels: int = 3
    width: int = 256
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class VideoEncoderConfig:
    """Temporal transformer aggregating per-frame tokens into a clip embedding."""

    width: int = 256
    depth: int = 2
    heads: int = 4
    mlp_ratio: float = 4.0
    max_frames: int = 16
    dropout: float = 0.0


@dataclass
class AudioEncoderConfig:
    """Spectrogram Transformer (AST-style) audio tower."""

    n_mels: int = 64
    max_frames: int = 200  # spectrogram time steps
    patch_time: int = 16
    patch_freq: int = 16
    width: int = 256
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class TextEncoderConfig:
    """Causal-free text transformer encoder."""

    vocab_size: int = 32000
    max_length: int = 32
    width: int = 256
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class FusionEncoderConfig:
    """Audio-video cross-attention fusion tower."""

    width: int = 256
    depth: int = 2
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Novel: correspondence gating strength (see docs/CONCEPTS.md).
    use_correspondence_gate: bool = True


@dataclass
class PEAVConfig:
    """Full PE-AV model configuration."""

    embed_dim: int = 256
    frame: FrameEncoderConfig = field(default_factory=FrameEncoderConfig)
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    fusion: FusionEncoderConfig = field(default_factory=FusionEncoderConfig)
    # Initial value of the learnable logit scale (log space): log(1 / 0.07).
    logit_scale_init: float = 2.6593
    logit_scale_max: float = 4.6052  # log(100)

    # ---- (de)serialisation helpers -------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PEAVConfig:
        data = dict(data)
        sub = {
            "frame": FrameEncoderConfig,
            "video": VideoEncoderConfig,
            "audio": AudioEncoderConfig,
            "text": TextEncoderConfig,
            "fusion": FusionEncoderConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in sub and isinstance(value, dict):
                kwargs[key] = sub[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PEAVConfig:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def preset(cls, name: str) -> PEAVConfig:
        if name not in PRESETS:
            raise KeyError(f"Unknown preset '{name}'. Choose from {sorted(PRESETS)}.")
        return cls.from_dict(PRESETS[name])


def _scaled_preset(embed_dim: int, width: int, frame_depth: int, audio_depth: int,
                   text_depth: int, video_depth: int, fusion_depth: int, heads: int) -> dict[str, Any]:
    """Build a preset dict where every tower shares the same hidden ``width``."""
    return {
        "embed_dim": embed_dim,
        "frame": {"width": width, "depth": frame_depth, "heads": heads},
        "video": {"width": width, "depth": video_depth, "heads": heads},
        "audio": {"width": width, "depth": audio_depth, "heads": heads},
        "text": {"width": width, "depth": text_depth, "heads": heads},
        "fusion": {"width": width, "depth": fusion_depth, "heads": heads},
    }


# Named presets loosely echoing the small / base / large release checkpoints.
PRESETS: dict[str, dict[str, Any]] = {
    "small": _scaled_preset(192, 192, 3, 3, 3, 2, 2, 3),
    "base": _scaled_preset(256, 256, 4, 4, 4, 2, 2, 4),
    "large": _scaled_preset(384, 384, 6, 6, 6, 3, 3, 6),
}
