"""A compact training loop for PE-AV.

The trainer wires the model, the multi-pair contrastive loss and the optional
PE-A-Frame frame-level loss together.  It is intentionally framework-light (no
Lightning / accelerate) so the optimisation logic is fully visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from ..losses import MultiPairContrastiveLoss, frame_level_contrastive
from ..modeling import PEAV


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 5
    batch_size: int = 32
    warmup_steps: int = 20
    grad_clip: float = 1.0
    frame_loss_weight: float = 0.5
    pair_weights: dict[str, float] = field(default_factory=dict)
    device: str = "cpu"
    log_every: int = 10


@dataclass
class TrainState:
    step: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    def __init__(self, model: PEAV, cfg: TrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.criterion = MultiPairContrastiveLoss(cfg.pair_weights)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.state = TrainState()

    def _lr_at(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / self.cfg.warmup_steps
        return self.cfg.lr

    def _embeddings(self, out) -> dict[str, torch.Tensor | None]:
        # Map model outputs onto the modality names used by the loss.
        return {
            "audio": out.audio_embed,
            "video": out.video_embed,
            "text": out.text_embed,
            "av": out.av_embed,
        }

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        device = self.cfg.device
        batch = {k: v.to(device) for k, v in batch.items()}

        out = self.model(
            frames=batch["frames"],
            spectrogram=batch["spectrogram"],
            time_mask=batch.get("time_mask"),
            token_ids=batch["token_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        result = self.criterion(self._embeddings(out), out.logit_scale)
        loss = result.total

        frame_loss = torch.tensor(0.0, device=device)
        if self.cfg.frame_loss_weight > 0 and out.frame_embeds is not None:
            frame_loss = frame_level_contrastive(
                out.frame_embeds, out.text_embed, out.logit_scale
            )
            loss = loss + self.cfg.frame_loss_weight * frame_loss

        lr = self._lr_at(self.state.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.model.clamp_logit_scale()
        self.state.step += 1

        logs = {"loss": float(loss.detach()), "frame_loss": float(frame_loss.detach()), "lr": lr}
        for key, value in result.per_pair.items():
            logs[f"loss/{key}"] = float(value.detach())
        return logs

    def fit(self, dataset, collate_fn=None) -> TrainState:
        collate_fn = collate_fn or getattr(dataset, "collate", None)
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate_fn
        )
        for epoch in range(self.cfg.epochs):
            for batch in loader:
                logs = self.train_step(batch)
                logs["epoch"] = epoch
                self.state.history.append(logs)
                if self.state.step % self.cfg.log_every == 0:
                    print(
                        f"epoch {epoch} step {self.state.step} "
                        f"loss {logs['loss']:.4f} frame {logs['frame_loss']:.4f} "
                        f"lr {logs['lr']:.2e}"
                    )
        return self.state
