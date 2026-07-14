import torch

from pe_av.losses import (
    ALL_PAIRS,
    MultiPairContrastiveLoss,
    frame_level_contrastive,
    info_nce,
)
from pe_av.modeling.common import l2_normalize


def test_all_pairs_count():
    # C(5, 2) = 10 pairwise objectives.
    assert len(ALL_PAIRS) == 10


def test_info_nce_perfect_alignment_is_low():
    torch.manual_seed(0)
    a = l2_normalize(torch.randn(16, 32))
    scale = torch.tensor(100.0)
    aligned = info_nce(a, a.clone(), scale)
    misaligned = info_nce(a, l2_normalize(torch.randn(16, 32)), scale)
    assert aligned < misaligned
    assert aligned >= 0


def test_multipair_only_uses_present_modalities():
    torch.manual_seed(0)
    emb = {
        "audio": l2_normalize(torch.randn(8, 16)),
        "video": l2_normalize(torch.randn(8, 16)),
        "text": None,
        "av": None,
    }
    loss = MultiPairContrastiveLoss()
    result = loss(emb, torch.tensor(10.0))
    assert set(result.per_pair) == {"audio__video"}
    assert torch.isfinite(result.total)


def test_multipair_weights_and_backward():
    torch.manual_seed(0)
    emb = {
        "audio": l2_normalize(torch.randn(8, 16, requires_grad=True)),
        "text": l2_normalize(torch.randn(8, 16, requires_grad=True)),
    }
    loss = MultiPairContrastiveLoss(weights={"audio__text": 2.0})
    result = loss(emb, torch.tensor(10.0))
    result.total.backward()
    assert "audio__text" in result.per_pair


def test_multipair_empty_is_differentiable_zero():
    scale = torch.tensor(10.0, requires_grad=True)
    result = MultiPairContrastiveLoss()({"audio": None}, scale)
    assert float(result.total.detach()) == 0.0
    result.total.backward()  # should not raise


def test_frame_level_contrastive():
    torch.manual_seed(0)
    frames = l2_normalize(torch.randn(6, 5, 16))
    text = l2_normalize(torch.randn(6, 16))
    loss = frame_level_contrastive(frames, text, torch.tensor(10.0))
    assert loss.ndim == 0 and torch.isfinite(loss)
