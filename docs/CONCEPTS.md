# Concepts & Theory

This companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) covers the *why*: the
contrastive learning theory PE-AV rests on, the reasoning behind the ten-pair
objective, and the **correspondence-gating** idea this repo contributes as a
novel-but-simple extension of the fusion tower.

---

## 1. Why contrastive learning gives you a shared space

Contrastive learning does not try to *reconstruct* anything. It only asks that
**matching things be close and non-matching things be far**. Formally, for a
batch of `N` matched pairs `(a_i, b_i)` with L2-normalised embeddings, InfoNCE
minimises

```
L = -1/N Σ_i log  exp(τ·a_iᵀb_i) / Σ_j exp(τ·a_iᵀb_j)
```

This is a softmax cross-entropy where the correct "class" for query `a_i` is its
true partner `b_i` among the `N` in-batch candidates. Two consequences matter:

1. **The temperature `τ` controls how hard the negatives bite.** Large `τ`
   sharpens the softmax and punishes near-misses; too large and training becomes
   unstable, which is why PE-AV (like CLIP) **clamps `τ ≤ 100`**.
2. **Negatives are free.** Every other item in the batch is a negative, so large
   batches give a better gradient signal. This is the single biggest reason
   contrastive pretraining scales so well.

Making the loss **symmetric** (average of `a→b` and `b→a`) removes a degenerate
solution where one modality collapses to predict the other.

---

## 2. Why *ten* pairwise objectives

PE-AV aligns five streams: `audio`, `video`, `av` (fused), `text`, and
`frame_text` (frame-level). Choosing 2 of 5 gives `C(5,2) = 10` pairs. Instead
of a single anchor modality (CLIP-style: everything ↔ text), PE-AV contrasts
**every pair it has labels for**.

Why this helps:

- **Transitivity for free.** If audio↔video and video↔text are both pulled tight,
  audio↔text alignment emerges even with *few* direct audio-text captions. The
  shared geometry propagates supervision across modalities.
- **Robustness to missing modalities.** Real web clips are unevenly labelled —
  some have speech transcripts, some have music tags, some have nothing but
  pixels and sound. A per-pair loss that **skips absent pairs** turns every
  partially-labelled sample into usable signal (see
  `MultiPairContrastiveLoss` — it only sums pairs present in the batch).
- **Caption-type scaling.** The paper's "data engine" synthesises *different
  kinds* of captions (speech content, musical attributes, sound-effect labels).
  Each caption type is just another `*_text` pair with its own weight.

The net effect the paper reports — and that you can reproduce at toy scale here —
is that **scaling cross-modality and caption-type pairs strengthens alignment**
and lifts zero-shot transfer.

---

## 3. The correspondence gate (this repo's novel twist)

**Problem.** Audio and video are not always in correspondence. A movie clip may
have a dubbed voice, off-screen narration, or a licensed music bed with nothing
to do with the picture. If the fusion tower *always* mixes audio into the joint
vector, these mismatches inject noise and destabilise retrieval on weakly-aligned
web data.

**Idea.** Before fusing, estimate how much the two streams actually agree, and
**gate the audio contribution by that agreement**. Concretely, from the
shared-space unimodal embeddings `a` and `v` we compute a scalar per sample:

```
g = sigmoid( w · cos(a, v) + b ),        w, b learnable
```

and scale the audio tokens entering the fusion transformer by `g` (see
`FusionEncoder.correspondence` and `FusionEncoder.forward`). Interpretation:

- **High agreement (`cos → 1`)** ⟹ `g → 1`: audio flows in fully, the fused
  vector is genuinely audiovisual.
- **Low / negative agreement** ⟹ `g → 0`: the gate closes and fusion falls back
  toward the **visual** stream, refusing to contaminate the joint embedding with
  unrelated sound.

Because `w` and `b` are learned, the model *discovers* how strict to be from the
data rather than us hard-coding a threshold. The gate is cheap (one dot product +
sigmoid), differentiable, and reduces to "always fuse" if the data are always
aligned (the model simply learns a large `w` and positive `b`).

This is a small, self-contained illustration of a general principle in
multimodal fusion: **condition the fusion on cross-modal reliability instead of
assuming it.** It is inspired by the alignment behaviour PE-AV needs at 100M-clip
scale, where a large fraction of pairs are only loosely corresponded.

---

## 4. Practical training notes

- **Batch size is a hyperparameter of the *loss*, not just memory.** More
  in-batch negatives ⇒ harder, more informative contrastive task.
- **Warmup + logit-scale clamp** stabilise early training when embeddings are
  still random and similarities are noisy (`training/trainer.py`).
- **L2-normalise before every comparison.** Forgetting this is the most common
  bug in contrastive code; here it is centralised in `l2_normalize` and applied
  inside every `encode_*` method so downstream code cannot forget.
- **Evaluate at the concept level.** With duplicated captions, exact-index
  recall understates quality; measuring whether the top hit shares the query's
  *concept* is the honest metric (see `tests/test_end_to_end.py`).

---

## 5. Further reading

- Meta AI — *Pushing the Frontier of Audiovisual Perception with Large-Scale
  Multimodal Correspondence Learning* (PE-AV),
  [arXiv:2512.19687](https://arxiv.org/abs/2512.19687).
- Meta AI — *Perception Encoder: The best visual embeddings are not at the output
  of the network* (the vision backbone PE-AV builds on).
- Radford et al. — *Learning Transferable Visual Models From Natural Language
  Supervision* (CLIP; the InfoNCE recipe).
- Gong et al. — *AST: Audio Spectrogram Transformer* (the audio tower design).
