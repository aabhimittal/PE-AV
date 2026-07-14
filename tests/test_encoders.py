import torch

from pe_av.config import (
    AudioEncoderConfig,
    FrameEncoderConfig,
    FusionEncoderConfig,
    TextEncoderConfig,
    VideoEncoderConfig,
)
from pe_av.modeling import (
    AudioEncoder,
    FrameEncoder,
    FusionEncoder,
    TextEncoder,
    VideoEncoder,
)


def test_frame_encoder_shapes():
    cfg = FrameEncoderConfig(image_size=64, patch_size=16, width=128, depth=2, heads=4)
    enc = FrameEncoder(cfg)
    cls, patches = enc(torch.randn(3, 3, 64, 64))
    assert cls.shape == (3, 128)
    assert patches.shape == (3, 16, 128)  # (64/16)^2 = 16 patches


def test_video_encoder_masking():
    cfg = VideoEncoderConfig(width=128, depth=2, heads=4, max_frames=8)
    enc = VideoEncoder(cfg, frame_dim=128)
    frames = torch.randn(2, 8, 128)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, 4:] = True  # second half of sample 0 is padding
    clip, tokens = enc(frames, mask)
    assert clip.shape == (2, 128)
    assert tokens.shape == (2, 8, 128)
    assert torch.isfinite(clip).all()


def test_audio_encoder_variable_length_padding():
    cfg = AudioEncoderConfig(n_mels=64, patch_time=16, patch_freq=16, width=128, depth=2, heads=4)
    enc = AudioEncoder(cfg)
    # T not a multiple of patch_time -> internal padding path exercised
    spec = torch.randn(2, 64, 100)
    time_mask = torch.zeros(2, 100, dtype=torch.bool)
    time_mask[1, 50:] = True
    cls, tokens = enc(spec, time_mask)
    assert cls.shape == (2, 128)
    assert torch.isfinite(cls).all()


def test_text_encoder_shapes():
    cfg = TextEncoderConfig(vocab_size=1000, max_length=16, width=128, depth=2, heads=4)
    enc = TextEncoder(cfg)
    ids = torch.randint(0, 1000, (4, 12))
    mask = torch.ones(4, 12, dtype=torch.long)
    cls, tokens = enc(ids, mask)
    assert cls.shape == (4, 128)
    assert tokens.shape == (4, 12, 128)


def test_fusion_correspondence_gate_in_range():
    cfg = FusionEncoderConfig(width=128, depth=2, heads=4, use_correspondence_gate=True)
    enc = FusionEncoder(cfg, audio_dim=128, video_dim=128)
    a = torch.randn(5, 128)
    v = torch.randn(5, 128)
    gate = enc.correspondence(a, v)
    assert gate.shape == (5, 1)
    assert ((gate >= 0) & (gate <= 1)).all()
    fused = enc(a, v)
    assert fused.shape == (5, 128)
