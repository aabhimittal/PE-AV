"""Datasets and batching for PE-AV.

``SyntheticAVDataset`` procedurally generates *correlated* audio, video and text
from a shared latent "concept" id.  Because the modalities genuinely share
information, contrastive training on this dataset actually converges and
retrieval accuracy climbs well above chance — making it a real end-to-end smoke
test of the whole pipeline, not just a shape check.

For real data, ``AudioVisualDataset`` shows the expected sample schema; plug in
your own decoding (e.g. ``torchvision`` / ``torchaudio``) in ``load_sample``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset

from .tokenizer import HashTokenizer
from .transforms import waveform_to_logmel

# A small vocabulary of "concepts", each with a caption and latent parameters
# that drive both the visual and the audio generators.
_CONCEPTS: list[dict] = [
    {"caption": "a dog barking loudly outdoors", "hz": 220.0, "hue": 0.02},
    {"caption": "a person playing acoustic guitar", "hz": 330.0, "hue": 0.14},
    {"caption": "ocean waves crashing on the beach", "hz": 110.0, "hue": 0.55},
    {"caption": "a car engine revving on a street", "hz": 165.0, "hue": 0.08},
    {"caption": "birds chirping in a quiet forest", "hz": 660.0, "hue": 0.33},
    {"caption": "rain falling on a metal roof", "hz": 440.0, "hue": 0.62},
    {"caption": "a baby laughing and giggling", "hz": 550.0, "hue": 0.90},
    {"caption": "a piano melody in a concert hall", "hz": 262.0, "hue": 0.75},
    {"caption": "footsteps walking on gravel", "hz": 130.0, "hue": 0.20},
    {"caption": "a crowd cheering at a stadium", "hz": 185.0, "hue": 0.98},
]


@dataclass
class AVSample:
    frames: torch.Tensor  # (T, C, H, W) in [0, 1]
    waveform: torch.Tensor  # (samples,)
    caption: str
    concept_id: int


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


class SyntheticAVDataset(Dataset):
    def __init__(
        self,
        length: int = 512,
        num_frames: int = 8,
        image_size: int = 64,
        sample_rate: int = 16000,
        clip_seconds: float = 1.0,
        n_mels: int = 64,
        max_time: int = 200,
        max_text_length: int = 32,
        seed: int = 0,
    ):
        self.length = length
        self.num_frames = num_frames
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.n_mels = n_mels
        self.max_time = max_time
        self.tokenizer = HashTokenizer(max_length=max_text_length)
        self.base_seed = seed

    def __len__(self) -> int:
        return self.length

    @property
    def num_concepts(self) -> int:
        return len(_CONCEPTS)

    def _make_sample(self, concept_id: int, gen: torch.Generator) -> AVSample:
        concept = _CONCEPTS[concept_id]
        size = self.image_size

        # --- video: a moving coloured bar whose colour encodes the concept ---
        r, g, b = _hsv_to_rgb(concept["hue"], 0.85, 0.9)
        base_color = torch.tensor([r, g, b]).view(3, 1, 1)
        frames = []
        for t in range(self.num_frames):
            canvas = torch.zeros(3, size, size)
            pos = int((t / max(self.num_frames - 1, 1)) * (size - size // 4))
            canvas[:, :, pos : pos + size // 4] = base_color
            canvas += 0.05 * torch.randn(3, size, size, generator=gen)
            frames.append(canvas.clamp(0, 1))
        video = torch.stack(frames)

        # --- audio: a tone at the concept frequency plus mild noise ---
        n = int(self.sample_rate * self.clip_seconds)
        time = torch.arange(n, dtype=torch.float32) / self.sample_rate
        hz = concept["hz"]
        wave = torch.sin(2 * math.pi * hz * time)
        wave = wave + 0.3 * torch.sin(2 * math.pi * 2 * hz * time)  # harmonic
        wave = wave + 0.05 * torch.randn(n, generator=gen)
        wave = wave / wave.abs().max().clamp_min(1e-6)

        return AVSample(video, wave, concept["caption"], concept_id)

    def __getitem__(self, idx: int) -> AVSample:
        gen = torch.Generator().manual_seed(self.base_seed + idx)
        concept_id = int(torch.randint(0, len(_CONCEPTS), (1,), generator=gen).item())
        return self._make_sample(concept_id, gen)

    def collate(self, batch: Sequence[AVSample]) -> dict[str, torch.Tensor]:
        return collate_av(batch, self.tokenizer, self.sample_rate, self.n_mels, self.max_time)


def collate_av(
    batch: Sequence[AVSample],
    tokenizer: HashTokenizer,
    sample_rate: int,
    n_mels: int,
    max_time: int,
) -> dict[str, torch.Tensor]:
    """Collate a list of :class:`AVSample` into padded model-ready tensors."""
    frames = torch.stack([b.frames for b in batch])  # (B, T, C, H, W)

    specs = [waveform_to_logmel(b.waveform, sample_rate, n_mels=n_mels) for b in batch]
    max_t = min(max(s.shape[-1] for s in specs), max_time)
    spec_batch = torch.zeros(len(batch), n_mels, max_t)
    time_mask = torch.ones(len(batch), max_t, dtype=torch.bool)
    for i, s in enumerate(specs):
        t = min(s.shape[-1], max_t)
        spec_batch[i, :, :t] = s[:, :t]
        time_mask[i, :t] = False  # real (unmasked) time steps

    tok = tokenizer([b.caption for b in batch])
    concept_ids = torch.tensor([b.concept_id for b in batch], dtype=torch.long)

    return {
        "frames": frames,
        "spectrogram": spec_batch,
        "time_mask": time_mask,
        "token_ids": tok.input_ids,
        "attention_mask": tok.attention_mask,
        "concept_ids": concept_ids,
    }


class AudioVisualDataset(Dataset):
    """Schema/adapter for real audio-video-caption data.

    ``records`` is a list of dicts; ``load_sample`` decodes one record into an
    :class:`AVSample`.  Override ``load_sample`` (or pass ``loader``) to hook up
    real video/audio decoding.
    """

    def __init__(self, records: list[dict], loader: Callable[[dict], AVSample] | None = None):
        self.records = records
        self.loader = loader

    def __len__(self) -> int:
        return len(self.records)

    def load_sample(self, record: dict) -> AVSample:
        if self.loader is None:
            raise NotImplementedError(
                "Provide a `loader` callable or subclass and implement `load_sample`."
            )
        return self.loader(record)

    def __getitem__(self, idx: int) -> AVSample:
        return self.load_sample(self.records[idx])
