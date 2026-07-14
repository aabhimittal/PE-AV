"""Contrastive objectives for PE-AV.

The core loss is the symmetric InfoNCE (CLIP) loss between two batches of
L2-normalised embeddings.  PE-AV's headline recipe is to apply this across
**ten pairwise modality combinations** and sum them; :class:`MultiPairContrastiveLoss`
implements exactly that with configurable per-pair weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn.functional as F

# The five embedding streams PE-AV aligns.  Choosing 2 of these 5 gives the
# C(5,2) = 10 pairwise contrastive objectives referenced in the paper.
MODALITIES = ("audio", "video", "text", "av", "frame_text")
ALL_PAIRS = tuple("__".join(sorted(p)) for p in combinations(MODALITIES, 2))


def info_nce(
    a: torch.Tensor,
    b: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE between paired, L2-normalised embeddings ``a`` and ``b``.

    Row ``i`` of ``a`` is the positive for row ``i`` of ``b`` and vice-versa; all
    other in-batch pairs are negatives.  ``logit_scale`` is the (already
    exponentiated) temperature.
    """
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes must match, got {a.shape} and {b.shape}.")
    logits = logit_scale * a @ b.t()  # (B, B)
    targets = torch.arange(a.shape[0], device=a.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)


@dataclass
class ContrastiveResult:
    total: torch.Tensor
    per_pair: dict[str, torch.Tensor]


class MultiPairContrastiveLoss(torch.nn.Module):
    """Sum of symmetric InfoNCE losses over every available modality pair.

    Parameters
    ----------
    weights:
        Optional mapping ``"<modA>__<modB>" -> float`` (keys sorted
        alphabetically, see :data:`ALL_PAIRS`).  Missing pairs default to 1.0.
        A pair is only included in the loss when *both* embeddings are present
        in the batch, so partially-labelled data works out of the box.
    """

    def __init__(self, weights: Mapping[str, float] | None = None):
        super().__init__()
        self.weights = dict(weights or {})

    @staticmethod
    def pair_key(mod_a: str, mod_b: str) -> str:
        return "__".join(sorted((mod_a, mod_b)))

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor | None],
        logit_scale: torch.Tensor,
        pairs: Iterable[tuple[str, str]] | None = None,
    ) -> ContrastiveResult:
        available = {k: v for k, v in embeddings.items() if v is not None}
        if pairs is None:
            pairs = [tuple(sorted(p)) for p in combinations(sorted(available), 2)]

        per_pair: dict[str, torch.Tensor] = {}
        total = None
        for mod_a, mod_b in pairs:
            if mod_a not in available or mod_b not in available:
                continue
            key = self.pair_key(mod_a, mod_b)
            weight = self.weights.get(key, 1.0)
            if weight == 0.0:
                continue
            loss = info_nce(available[mod_a], available[mod_b], logit_scale)
            per_pair[key] = loss
            total = weight * loss if total is None else total + weight * loss

        if total is None:
            # No valid pairs: return a differentiable zero anchored to logit_scale.
            total = logit_scale.sum() * 0.0
        return ContrastiveResult(total=total, per_pair=per_pair)


def frame_level_contrastive(
    frame_embeds: torch.Tensor,
    text_embed: torch.Tensor,
    logit_scale: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PE-A-Frame objective: align each *frame* with the clip's text embedding.

    This is a fine-grained (max-over-time) alignment used for audio-frame-to-text
    tasks such as sound-event detection.  ``frame_embeds`` is ``(B, T, D)`` and
    ``text_embed`` is ``(B, D)``.  For each (clip, caption) pair we take the
    best-matching frame (max over time) and run a symmetric InfoNCE over the
    batch of these max-pooled similarities.
    """
    b, t, d = frame_embeds.shape
    # sim[i, j, k] = similarity of clip i's frame k with caption j.
    sim = torch.einsum("itd,jd->ijt", frame_embeds, text_embed)
    if frame_mask is not None:
        sim = sim.masked_fill(frame_mask[:, None, :], float("-inf"))
    clip_text_sim = sim.max(dim=-1).values  # (B, B): max-over-frames
    logits = logit_scale * clip_text_sim
    targets = torch.arange(b, device=frame_embeds.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
