#!/usr/bin/env python3
"""Extract shared-space PE-AV embeddings for a batch of samples and save them.

This mirrors how you would build a retrieval index over a real corpus: encode
each modality, then persist the L2-normalised vectors.  Here we use the
synthetic dataset as a stand-in corpus.

Usage:
    python scripts/extract_embeddings.py --preset small --n 128 --out embeds.pt
"""

from __future__ import annotations

import argparse

import torch

from pe_av import PEAV, PEAVConfig
from pe_av.data import SyntheticAVDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--out", default="embeds.pt")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = PEAVConfig.from_dict(ckpt["config"])
        model = PEAV(cfg)
        model.load_state_dict(ckpt["state_dict"])
    else:
        cfg = PEAVConfig.preset(args.preset)
        model = PEAV(cfg)
    model.eval()

    ds = SyntheticAVDataset(length=args.n, num_frames=cfg.video.max_frames // 2,
                            image_size=cfg.frame.image_size, n_mels=cfg.audio.n_mels)
    batch = ds.collate([ds[i] for i in range(len(ds))])
    with torch.no_grad():
        out = model(frames=batch["frames"], spectrogram=batch["spectrogram"],
                    time_mask=batch["time_mask"], token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"])

    torch.save(
        {
            "audio": out.audio_embed,
            "video": out.video_embed,
            "text": out.text_embed,
            "av": out.av_embed,
            "captions": [ds[i].caption for i in range(len(ds))],
            "concept_ids": batch["concept_ids"],
        },
        args.out,
    )
    print(f"Saved {args.n} embeddings ({cfg.embed_dim}-d) to {args.out}")


if __name__ == "__main__":
    main()
