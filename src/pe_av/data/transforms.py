"""Audio and video preprocessing.

The mel filterbank is implemented from scratch (Slaney-style triangular filters)
so the package has no ``librosa`` / ``torchaudio`` dependency and stays trivially
installable.  ``waveform_to_logmel`` is the reference audio front-end;
``sample_frames`` handles temporal frame sampling for video clips.
"""

from __future__ import annotations

import math

import torch


def hz_to_mel(hz: torch.Tensor | float) -> torch.Tensor | float:
    return 2595.0 * math.log10(1.0 + (hz if isinstance(hz, float) else hz) / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, f_min: float = 0.0, f_max: float | None = None
) -> torch.Tensor:
    """Return a ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank matrix."""
    f_max = f_max or sample_rate / 2
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    m_min, m_max = hz_to_mel(float(f_min)), hz_to_mel(float(f_max))
    m_points = torch.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = f_points[m - 1], f_points[m], f_points[m + 1]
        up = (all_freqs - left) / (center - left).clamp_min(1e-6)
        down = (right - all_freqs) / (right - center).clamp_min(1e-6)
        fb[m - 1] = torch.maximum(torch.zeros_like(all_freqs), torch.minimum(up, down))
    return fb


def waveform_to_logmel(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert a mono ``waveform`` ``(...,samples)`` to a log-mel ``(...,n_mels,T)``."""
    window = torch.hann_window(n_fft, device=waveform.device)
    spec = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=True,
    )
    power = spec.abs() ** 2  # (..., n_freqs, T)
    fb = mel_filterbank(n_mels, n_fft, sample_rate).to(waveform.device)
    mel = torch.matmul(fb, power)  # (..., n_mels, T)
    logmel = torch.log(mel + eps)
    # Per-example standardisation keeps the encoder input well-conditioned.
    logmel = (logmel - logmel.mean(dim=(-2, -1), keepdim=True)) / (
        logmel.std(dim=(-2, -1), keepdim=True) + eps
    )
    return logmel


def sample_frames(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Uniformly sample (or pad) ``num_frames`` from ``video`` ``(T, C, H, W)``."""
    t = video.shape[0]
    if t == num_frames:
        return video
    if t > num_frames:
        idx = torch.linspace(0, t - 1, num_frames).round().long()
        return video[idx]
    # Too short: repeat the last frame to pad.
    pad = video[-1:].expand(num_frames - t, *video.shape[1:])
    return torch.cat([video, pad], dim=0)
