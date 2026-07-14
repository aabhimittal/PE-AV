# PE-AV Architecture — a step-by-step walkthrough

This document explains **how PE-AV works, from raw signals to a shared embedding
space**, and how this repository implements each stage. It is meant to be read
top-to-bottom alongside the code in `src/pe_av/`.

> **Scope.** This is an independent, educational re-implementation of the ideas
> in Meta AI's *Perception Encoder Audiovisual* (PE-AV,
> [arXiv:2512.19687](https://arxiv.org/abs/2512.19687)). It reproduces the
> **architecture and training recipe** at small scale on synthetic data — it is
> **not** the official weights and does not download the 100M-pair corpus.

---

## 0. The one-sentence idea

> Learn **one** embedding space where a clip's audio, its video, a joint
> audiovisual vector, and a text caption all land in the **same place** — so that
> comparing *any* two modalities is a single dot product.

Everything else is engineering in service of that sentence.

```
                                   ┌─────────────────────────┐
   video frames ─▶ Frame ─▶ Video ─┤                         │
                                   │   shared embedding       │
   waveform ────▶ Mel ──▶ Audio ───┤   space  (L2-normalised, │──▶  retrieval,
                          │        │   one temperature τ)     │     SAM Audio,
                          ▼        │                         │     zero-shot
   (audio,video) ─▶ AV-Fusion ─────┤                         │
                                   │                         │
   caption ─────────────▶ Text ────┤                         │
                                   └─────────────────────────┘
```

---

## 1. The five towers

PE-AV factorises perception into **five encoders**, each ending in a linear
projection into the shared `embed_dim` space (`src/pe_av/modeling/`):

| Tower | Input | File | Output |
|-------|-------|------|--------|
| **Frame** | one RGB frame `(C,H,W)` | `frame_encoder.py` | per-frame token |
| **Video** | sequence of frame tokens | `video_encoder.py` | clip vector + frame tokens |
| **Audio** | log-mel spectrogram `(n_mels,T)` | `audio_encoder.py` | audio vector |
| **Fusion** | audio + video tokens | `fusion_encoder.py` | joint AV vector |
| **Text** | token ids | `text_encoder.py` | caption vector |

Why split *frame* and *video*? The per-frame ViT is the expensive part. By
computing frame features once and letting a small temporal transformer aggregate
them, you can (a) share the image backbone with an image model and (b) cheaply
trade off temporal resolution. This is the same "frame encoder + video encoder"
split named in the PE-AV release.

### 1.1 Frame encoder (Vision Transformer)

```
frame (3×64×64) ─▶ 16×16 conv patches ─▶ +[CLS] ─▶ +pos ─▶ Transformer ─▶ LayerNorm
                                                                    │
                                                     [CLS] token = frame summary
```

Standard ViT: patchify with a strided conv, prepend a learnable `[CLS]`, add
learned positional embeddings, run pre-norm transformer blocks. We return **both**
the `[CLS]` summary and the patch tokens.

### 1.2 Video encoder (temporal transformer)

The frame tower is applied to every frame independently (batched as `B·T`), then:

```
frame summaries (B,T,D) ─▶ +[VID] token ─▶ + sinusoidal temporal pos
                        ─▶ Transformer (with padding mask) ─▶ LayerNorm
   [VID] token = clip embedding      frame tokens = temporally-contextualised frames
```

The frame tokens it returns are what **PE-A-Frame** aligns to text for
frame-level localisation (§4).

### 1.3 Audio encoder (Audio Spectrogram Transformer)

Audio is first turned into a **log-mel spectrogram** (`data/transforms.py`, a
from-scratch mel filterbank — no `torchaudio` needed). The AST then treats the
`(n_mels, T)` image exactly like the ViT treats an RGB image, but with
**rectangular time×frequency patches**:

```
log-mel (64×T) ─▶ (16 freq × 16 time) conv patches ─▶ +[CLS] ─▶ +pos
               ─▶ Transformer (with time padding mask) ─▶ LayerNorm ─▶ [CLS] = audio vector
```

Variable-length clips are handled two ways: the time axis is right-padded to a
multiple of the patch size, and a **time padding mask** is collapsed to
per-patch and fed to attention so padded regions are ignored.

### 1.4 Text encoder

A bidirectional transformer over token ids with a prepended `[CLS]`. Captions in
the PE-AV data engine describe **speech, music and sound-effects**; the pooled
`[CLS]` is the text embedding used for audio↔text and video↔text alignment.

### 1.5 Fusion encoder (the joint AV vector)

See [`CONCEPTS.md`](CONCEPTS.md) for the **correspondence-gating** detail. In
short: project audio and video tokens to a common width, tag them with modality
type embeddings, prepend a `[FUSE]` token, and run a cross-modal transformer. The
`[FUSE]` token becomes the joint audiovisual embedding — the vector SAM Audio
consumes to know *what to listen for*.

---

## 2. The shared space and the temperature

Each tower's output is passed through a bias-free linear **projection head** and
then **L2-normalised** (`modeling/pe_av.py`). After normalisation, a dot product
*is* cosine similarity, and cosine similarity across towers is only meaningful
because they all project into the **same** `embed_dim`.

Similarities are scaled by a single **learnable temperature** `τ` (stored in log
space as `logit_scale`, clamped to ≤ log 100, exactly as in CLIP). One global
temperature keeps all ten modality pairs on a comparable scale.

---

## 3. Training objective — ten pairwise contrastive losses

With five embedding streams, there are `C(5,2) = 10` modality pairs. PE-AV's
headline recipe is to apply a **symmetric InfoNCE (CLIP) loss to every available
pair** and sum them (`losses/contrastive.py`):

```
total = Σ_over_pairs  w[pair] · InfoNCE(embed_a, embed_b; τ)
```

Key implementation choices:

- **InfoNCE** (`info_nce`): for a batch of matched pairs, row *i* of stream A is
  the positive for row *i* of stream B; all other in-batch items are negatives.
  The loss is symmetric (A→B and B→A cross-entropy averaged).
- **Only present pairs contribute** (`MultiPairContrastiveLoss`): if a batch has
  no text, the text pairs are silently skipped. This lets you mix fully- and
  partially-labelled data — exactly what "scaling caption types" needs.
- **Per-pair weights** let you emphasise, e.g., audio↔text over video↔text.

This is *why* the model in `examples/quickstart.py` learns any-to-any retrieval
from a single training run: every batch tightens all pairs at once, and
transitivity (audio≈video, video≈text ⟹ audio≈text) does the rest.

---

## 4. PE-A-Frame — fine-grained frame-level alignment

The main objective aligns **clip-level** vectors. For tasks like **sound-event
detection** you need to know *when* something happens. PE-A-Frame fine-tunes with
a **frame-level contrastive loss** (`frame_level_contrastive`): each caption is
matched to the **best-matching frame** (max-over-time) of its clip, so gradients
push individual frames — not just the clip average — toward the text. Our trainer
adds this as an auxiliary loss with weight `frame_loss_weight`.

---

## 5. From embeddings to applications

Because everything is one normalised space, downstream use is trivial
(`retrieval/index.py`):

- **Cross-modal retrieval:** encode a text query, dot-product against a bank of
  AV embeddings, take top-k. Same code retrieves audio-from-video,
  video-from-audio, etc.
- **Zero-shot classification:** embed class-name prompts, pick the nearest.
- **SAM Audio conditioning:** the joint AV / audio embedding is the prompt that
  tells a separation model which source to isolate.

See [`CONCEPTS.md`](CONCEPTS.md) for the theory behind the contrastive objective
and the correspondence gate, and the top-level `README.md` for runnable commands.
