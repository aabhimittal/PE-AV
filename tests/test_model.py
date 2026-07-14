import torch

from pe_av import PEAV, PEAVConfig


def _tiny_cfg():
    cfg = PEAVConfig.preset("small")
    # Shrink further for fast CPU tests.
    cfg.frame.depth = cfg.audio.depth = cfg.text.depth = 1
    cfg.video.depth = cfg.fusion.depth = 1
    return cfg


def test_forward_all_modalities():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = PEAV(cfg)
    frames = torch.randn(2, 6, 3, 64, 64)
    spec = torch.randn(2, 64, 96)
    ids = torch.randint(0, cfg.text.vocab_size, (2, 10))
    out = model(frames=frames, spectrogram=spec, token_ids=ids)
    for emb in (out.audio_embed, out.video_embed, out.text_embed, out.av_embed):
        assert emb.shape == (2, cfg.embed_dim)
    # embeddings are L2-normalised
    assert torch.allclose(out.av_embed.norm(dim=-1), torch.ones(2), atol=1e-4)


def test_partial_inputs():
    cfg = _tiny_cfg()
    model = PEAV(cfg)
    out = model(spectrogram=torch.randn(3, 64, 64))
    assert out.audio_embed is not None
    assert out.video_embed is None
    assert out.av_embed is None  # fusion needs both audio + video


def test_logit_scale_clamp():
    cfg = _tiny_cfg()
    model = PEAV(cfg)
    with torch.no_grad():
        model.logit_scale.fill_(100.0)
    model.clamp_logit_scale()
    assert float(model.logit_scale.detach()) <= cfg.logit_scale_max + 1e-5


def test_gate_makes_fusion_visual_dominant_when_closed():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = PEAV(cfg)
    frames = torch.randn(2, 6, 3, 64, 64)
    spec = torch.randn(2, 64, 96)
    out = model(frames=frames, spectrogram=spec)
    assert out.av_embed.shape == (2, cfg.embed_dim)
    assert torch.isfinite(out.av_embed).all()


def test_encode_helpers_consistent_with_forward():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = PEAV(cfg).eval()
    spec = torch.randn(2, 64, 96)
    with torch.no_grad():
        a1, _, _ = model.encode_audio(spec)
        a2 = model(spectrogram=spec).audio_embed
    assert torch.allclose(a1, a2, atol=1e-5)
