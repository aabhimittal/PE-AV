"""End-to-end learning test: the model must beat chance after a short train run.

This is the strongest correctness signal in the suite — it exercises the data
engine, all five towers, the multi-pair loss, the optimiser and the retrieval
index together, and asserts that cross-modal alignment genuinely emerges.
"""

import torch

from pe_av import PEAV, PEAVConfig
from pe_av.data import SyntheticAVDataset
from pe_av.training import TrainConfig, Trainer


def test_training_reduces_loss_and_learns_alignment():
    torch.manual_seed(0)
    cfg = PEAVConfig.preset("small")
    model = PEAV(cfg)
    dataset = SyntheticAVDataset(length=256, num_frames=8, image_size=64, n_mels=64)

    trainer = Trainer(model, TrainConfig(epochs=4, batch_size=64, lr=5e-4, log_every=1000))
    state = trainer.fit(dataset)

    first = state.history[0]["loss"]
    last = state.history[-1]["loss"]
    assert last < first  # loss actually goes down

    # Concept-level cross-modal retrieval should be well above chance (0.1).
    model.eval()
    ev = SyntheticAVDataset(length=120, num_frames=8, image_size=64, n_mels=64, seed=7)
    batch = ev.collate([ev[i] for i in range(len(ev))])
    concepts = batch["concept_ids"]
    with torch.no_grad():
        out = model(frames=batch["frames"], spectrogram=batch["spectrogram"],
                    time_mask=batch["time_mask"], token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"])

    sims = out.text_embed @ out.av_embed.t()
    sims.fill_diagonal_(-1e9)
    top = sims.argmax(-1)
    acc = (concepts[top] == concepts).float().mean().item()
    assert acc > 0.4  # chance is 0.1; alignment clears it comfortably
