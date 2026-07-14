#!/usr/bin/env python3
"""End-to-end PE-AV quickstart: train briefly, then run any-to-any retrieval.

Run with:  python examples/quickstart.py
"""

from __future__ import annotations

import torch

from pe_av import PEAV, PEAVConfig
from pe_av.data import SyntheticAVDataset
from pe_av.retrieval import EmbeddingIndex
from pe_av.training import TrainConfig, Trainer


def main() -> None:
    torch.manual_seed(0)

    cfg = PEAVConfig.preset("small")
    model = PEAV(cfg)
    print(f"PE-AV/small — {model.num_parameters() / 1e6:.1f}M parameters")

    # 1. Train on procedurally-correlated audio / video / text.
    dataset = SyntheticAVDataset(length=512, num_frames=8, image_size=64, n_mels=64)
    trainer = Trainer(model, TrainConfig(epochs=5, batch_size=64, lr=5e-4, log_every=20))
    trainer.fit(dataset)

    # 2. Encode a held-out gallery of audiovisual clips into the shared space.
    model.eval()
    gallery = SyntheticAVDataset(length=100, num_frames=8, image_size=64, n_mels=64, seed=123)
    batch = gallery.collate([gallery[i] for i in range(len(gallery))])
    with torch.no_grad():
        out = model(frames=batch["frames"], spectrogram=batch["spectrogram"],
                    time_mask=batch["time_mask"])

    index = EmbeddingIndex(dim=cfg.embed_dim)
    index.add(out.av_embed, [gallery[i].caption for i in range(len(gallery))])

    # 3. Retrieve audiovisual clips from a free-text query.
    query = "birds chirping in a quiet forest"
    tok = gallery.tokenizer(query)
    with torch.no_grad():
        text_embed = model.encode_text(tok.input_ids, tok.attention_mask)

    print(f'\nQuery: "{query}"')
    for rank, hit in enumerate(index.search(text_embed, top_k=5)[0], 1):
        print(f"  {rank}. ({hit.score:.3f}) {hit.payload}")


if __name__ == "__main__":
    main()
