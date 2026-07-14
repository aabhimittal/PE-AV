# PE-AV — Perception Encoder Audiovisual

> A from-scratch, **runnable** reimplementation of the ideas behind Meta AI's
> **Perception Encoder Audiovisual (PE-AV)** — the audiovisual encoder powering
> **SAM Audio** and large-scale multimodal retrieval.

[![CI](https://github.com/aabhimittal/pe-av/actions/workflows/ci.yml/badge.svg)](https://github.com/aabhimittal/pe-av/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

PE-AV learns a **single embedding space** shared by five streams — a video's
**frames**, the **clip**, the **audio**, a fused **audiovisual** vector, and a
**text** caption — using **scaled contrastive learning** over *ten* pairwise
objectives. Once trained, comparing any two modalities (audio↔text, video↔audio,
text↔audiovisual, …) is a single dot product, which is what makes cross-modal
retrieval and SAM-Audio-style sound prompting possible.

This repository reproduces that **architecture and training recipe at small
scale**, with a real end-to-end pipeline you can run on a laptop CPU in minutes:
five transformer towers, the multi-pair InfoNCE objective, a PE-A-Frame
frame-level loss, a from-scratch mel front-end, a synthetic *correlated* AV data
engine, a training loop, and a cross-modal retrieval index — all tested.

> **What this is / isn't.** An **independent, educational** implementation of the
> concepts in [arXiv:2512.19687](https://arxiv.org/abs/2512.19687). It is **not**
> Meta's official weights and does not ship the 100M-pair corpus. It is built to
> *teach the method* and *prove it works* — the model genuinely learns any-to-any
> alignment on the included synthetic data (see [Results](#results)).

---

## Table of contents
- [Why PE-AV](#why-pe-av)
- [Install](#install)
- [Quickstart](#quickstart-60-seconds)
- [How it works](#how-it-works-conceptual-tour)
- [The novel bit: correspondence gating](#the-novel-bit-correspondence-gating)
- [Results](#results)
- [Model presets](#model-presets)
- [Project layout](#project-layout)
- [CLI & scripts](#cli--scripts)
- [Testing](#testing)
- [Citation & credits](#citation--credits)

---

## Why PE-AV

Most multimodal encoders align **one** pair (e.g. image↔text in CLIP). Audio and
video in the wild are richer and messier: speech, music and sound effects; frames
that may or may not correspond to the sound. PE-AV's contributions, mirrored
here, are:

1. **One shared space for 5 streams**, so *any* modality retrieves *any* other.
2. **Ten pairwise contrastive objectives** that share supervision transitively
   and tolerate partially-labelled data.
3. **A joint audiovisual (fusion) embedding** — the vector a separation model
   like SAM Audio uses to know *which sound to isolate*.
4. **PE-A-Frame**: a frame-level objective for temporal tasks like sound-event
   detection.

---

## Install

```bash
git clone https://github.com/aabhimittal/pe-av.git
cd pe-av

# CPU-only PyTorch (skip if you already have torch):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# The package + dev/test extras:
pip install -e ".[dev]"
```

Python ≥ 3.9. The only runtime deps are `torch`, `numpy`, `pyyaml` — the mel
filterbank and tokenizer are implemented from scratch, so there is **no
`torchaudio` / `librosa` / tokenizer download** to fight with.

---

## Quickstart (60 seconds)

```bash
python examples/quickstart.py
```

This trains PE-AV/small on the synthetic AV data engine and then retrieves
audiovisual clips from a free-text query:

```
PE-AV/small — 12.4M parameters
...
Query: "birds chirping in a quiet forest"
  1. (0.857) birds chirping in a quiet forest
  2. (0.849) birds chirping in a quiet forest
  ...
```

Or in Python:

```python
import torch
from pe_av import PEAV, PEAVConfig

model = PEAV(PEAVConfig.preset("small"))

frames = torch.randn(2, 8, 3, 64, 64)   # (B, T, C, H, W)
spec   = torch.randn(2, 64, 128)        # (B, n_mels, time)
tokens = torch.randint(0, 32000, (2, 12))

out = model(frames=frames, spectrogram=spec, token_ids=tokens)
out.audio_embed.shape   # (2, 192)  — all embeddings share one space
out.video_embed.shape   # (2, 192)
out.text_embed.shape    # (2, 192)
out.av_embed.shape      # (2, 192)  — joint audiovisual vector
out.frame_embeds.shape  # (2, 8, 192) — per-frame, for PE-A-Frame localisation

# any-to-any similarity is just a dot product:
sim_audio_text = out.audio_embed @ out.text_embed.T
```

---

## How it works (conceptual tour)

Five towers each project into a shared, L2-normalised space compared with one
learnable temperature:

```
   video frames ─▶ Frame(ViT) ─▶ Video(temporal) ─┐
                                                   ├─▶  shared space  ─▶ retrieval /
   waveform ─▶ log-mel ─▶ Audio(AST) ──────────────┤   (dot product)     SAM Audio /
                              └──▶ AV-Fusion ───────┤                     zero-shot
   caption ─▶ Text(transformer) ────────────────────┘
```

- **Frame** — a Vision Transformer over 16×16 patches of each RGB frame.
- **Video** — a temporal transformer aggregating frame tokens into a clip vector
  (and returning contextualised per-frame tokens for PE-A-Frame).
- **Audio** — an Audio Spectrogram Transformer over time×frequency patches of a
  log-mel spectrogram, with a time padding mask for variable-length clips.
- **Fusion** — cross-attends audio + video into a joint AV vector (with the
  correspondence gate below).
- **Text** — a bidirectional transformer over caption tokens.

**Training** applies a symmetric InfoNCE (CLIP) loss to **every available
modality pair** — `C(5,2) = 10` of them — plus an optional PE-A-Frame
frame-level loss. Because a batch tightens all pairs at once, alignment
propagates transitively (audio≈video, video≈text ⟹ audio≈text), so a single
run yields **any-to-any** retrieval.

📖 Full walkthrough: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** ·
Theory & design rationale: **[docs/CONCEPTS.md](docs/CONCEPTS.md)**

---

## The novel bit: correspondence gating

Natural audio and video only *sometimes* correspond (dubbing, off-screen
narration, background music). Instead of blindly fusing, the fusion tower
estimates per-sample audio-video agreement and **gates** how much audio enters
the joint embedding:

```
g = sigmoid( w · cos(audio_embed, video_embed) + b )      # learnable w, b
```

When the streams agree, `g→1` and the fused vector is genuinely audiovisual; when
they don't, `g→0` and fusion falls back toward the visual stream instead of
contaminating the joint embedding with unrelated sound. It's one dot product and
a sigmoid, fully differentiable, and reduces to "always fuse" if the data are
always aligned. Derivation in [docs/CONCEPTS.md §3](docs/CONCEPTS.md).

---

## Results

Trained on the synthetic AV data engine (10 concepts, correlated audio/video/text),
PE-AV/small learns cross-modal alignment far above chance in **a handful of CPU
epochs**. Concept-level top-1 retrieval on a held-out gallery (chance = 0.10):

| Query → Gallery      | top-1 accuracy |
|----------------------|:--------------:|
| text → audiovisual   | **1.00**       |
| text → audio         | **1.00**       |
| audio → video        | **0.76**       |
| _chance_             | 0.10           |

Reproduce with `python examples/quickstart.py` and
`tests/test_end_to_end.py` (which asserts alignment clears chance). Numbers are
illustrative of the *method*, not the paper's large-scale benchmarks.

---

## Model presets

| Preset | embed dim | width | depth (frame/audio/text) | params |
|--------|:---------:|:-----:|:------------------------:|:------:|
| `small` | 192 | 192 | 3 / 3 / 3 | 12.4M |
| `base`  | 256 | 256 | 4 / 4 / 4 | 21.6M |
| `large` | 384 | 384 | 6 / 6 / 6 | 56.3M |

```python
model = PEAV.from_preset("base")            # or
cfg   = PEAVConfig.from_yaml("configs/pe_av_base.yaml")
```

Everything is config-driven (`configs/*.yaml`, `pe_av.config.PEAVConfig`), so you
can resize any tower independently.

---

## Project layout

```
src/pe_av/
├── config.py              # dataclass configs + YAML + presets
├── modeling/
│   ├── common.py          # attention / transformer building blocks
│   ├── frame_encoder.py   # ViT image tower
│   ├── video_encoder.py   # temporal aggregation
│   ├── audio_encoder.py   # Audio Spectrogram Transformer
│   ├── fusion_encoder.py  # AV fusion + correspondence gate
│   ├── text_encoder.py    # caption transformer
│   └── pe_av.py           # the full model + shared projections
├── losses/contrastive.py  # InfoNCE, 10-pair loss, PE-A-Frame loss
├── data/                  # mel transform, tokenizer, synthetic AV dataset
├── training/trainer.py    # training loop (warmup, clamp, frame loss)
├── retrieval/index.py     # shared-space embedding index + recall@k
└── cli.py                 # pe-av-train / pe-av-retrieve entry points
docs/    ARCHITECTURE.md · CONCEPTS.md
scripts/ train.py · retrieval_demo.py · extract_embeddings.py
examples/quickstart.py
tests/   26 tests incl. an end-to-end learning test
```

---

## CLI & scripts

```bash
# Train and save a checkpoint
pe-av-train --preset small --epochs 5 --save checkpoints/pe_av.pt
#   (equivalently: python scripts/train.py ...)

# Text -> audiovisual retrieval demo
pe-av-retrieve --checkpoint checkpoints/pe_av.pt

# Dump shared-space embeddings for an external index
python scripts/extract_embeddings.py --preset small --n 128 --out embeds.pt
```

### Using your own data

Subclass `AudioVisualDataset` (or pass a `loader`) to yield `AVSample(frames,
waveform, caption, concept_id)`; the provided `collate_av` turns waveforms into
log-mel spectrograms and tokenises captions. See `src/pe_av/data/dataset.py`.

---

## Testing

```bash
pytest -q          # 26 tests: encoders, losses, model, retrieval, end-to-end
ruff check .       # lint
```

The end-to-end test actually trains a model and asserts that cross-modal
retrieval beats chance — so CI fails if the learning pipeline regresses, not just
the shapes.

---

## Citation & credits

This project reimplements ideas from Meta AI's PE-AV. If you use those ideas,
please cite the original work:

```bibtex
@article{peav2025,
  title   = {Pushing the Frontier of Audiovisual Perception with
             Large-Scale Multimodal Correspondence Learning},
  author  = {{Meta AI}},
  journal = {arXiv preprint arXiv:2512.19687},
  year    = {2025}
}
```

- Original PE-AV paper: <https://arxiv.org/abs/2512.19687>
- Perception Encoder (vision backbone): Meta AI, 2025
- SAM Audio: <https://ai.meta.com/blog/sam-audio/>
- Official weights: `facebook/pe-av-large` on Hugging Face

Licensed under the [MIT License](LICENSE). This is an unofficial, independent
educational reimplementation and is not affiliated with or endorsed by Meta.
