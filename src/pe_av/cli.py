"""Console-script entry points (installed as ``pe-av-train`` / ``pe-av-retrieve``)."""

from __future__ import annotations

import argparse

import torch

from .config import PEAVConfig
from .data import SyntheticAVDataset
from .modeling import PEAV
from .retrieval import EmbeddingIndex
from .training import TrainConfig, Trainer


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PE-AV on the synthetic AV dataset.")
    parser.add_argument("--preset", default="small", choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default=None, help="Path to save the trained checkpoint.")
    args = parser.parse_args(argv)

    cfg = PEAVConfig.preset(args.preset)
    model = PEAV(cfg)
    print(f"PE-AV/{args.preset}: {model.num_parameters() / 1e6:.2f}M parameters")

    dataset = SyntheticAVDataset(
        length=args.dataset_size,
        num_frames=cfg.video.max_frames // 2,
        image_size=cfg.frame.image_size,
        n_mels=cfg.audio.n_mels,
    )
    trainer = Trainer(
        model,
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device),
    )
    trainer.fit(dataset)

    if args.save:
        torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()}, args.save)
        print(f"Saved checkpoint to {args.save}")


def retrieve_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a text->audiovisual retrieval demo.")
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint.")
    parser.add_argument("--preset", default="small", choices=["small", "base", "large"])
    parser.add_argument("--gallery-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = PEAVConfig.from_dict(ckpt["config"])
        model = PEAV(cfg)
        model.load_state_dict(ckpt["state_dict"])
    else:
        cfg = PEAVConfig.preset(args.preset)
        model = PEAV(cfg)
    model.eval()

    dataset = SyntheticAVDataset(
        length=args.gallery_size,
        num_frames=cfg.video.max_frames // 2,
        image_size=cfg.frame.image_size,
        n_mels=cfg.audio.n_mels,
    )
    batch = dataset.collate([dataset[i] for i in range(len(dataset))])

    index = EmbeddingIndex(dim=cfg.embed_dim)
    with torch.no_grad():
        out = model(
            frames=batch["frames"], spectrogram=batch["spectrogram"],
            time_mask=batch["time_mask"],
        )
        index.add(out.av_embed, [dataset[i].caption for i in range(len(dataset))])

        query = "birds chirping in a quiet forest"
        text = model.encode_text(*_encode_query(dataset, query))
    hits = index.search(text, top_k=args.top_k)[0]
    print(f'Query: "{query}"')
    for rank, hit in enumerate(hits, 1):
        print(f"  {rank}. ({hit.score:.3f}) {hit.payload}")


def _encode_query(dataset: SyntheticAVDataset, text: str):
    tok = dataset.tokenizer(text)
    return tok.input_ids, tok.attention_mask


if __name__ == "__main__":
    train_main()
