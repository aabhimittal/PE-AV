"""A minimal in-memory embedding index for cross-modal retrieval.

Because every PE-AV embedding is L2-normalised and lives in one shared space,
retrieval is just a matrix product followed by top-k.  This class stores a bank
of embeddings with arbitrary payloads and answers "given this query vector, what
are the nearest items?" — regardless of which modality produced either side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..modeling.common import l2_normalize


@dataclass
class RetrievalHit:
    index: int
    score: float
    payload: Any


@dataclass
class EmbeddingIndex:
    dim: int
    _vectors: list[torch.Tensor] = field(default_factory=list)
    _payloads: list[Any] = field(default_factory=list)
    _matrix: torch.Tensor | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self._payloads)

    def add(self, vectors: torch.Tensor, payloads: Sequence[Any]) -> None:
        """Add a batch of ``(N, dim)`` embeddings with matching ``payloads``."""
        vectors = vectors.detach().float()
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}.")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("Number of payloads must match number of vectors.")
        vectors = l2_normalize(vectors)
        self._vectors.extend(vectors)
        self._payloads.extend(payloads)
        self._matrix = None  # invalidate cache

    def _build(self) -> torch.Tensor:
        if self._matrix is None:
            if not self._vectors:
                raise RuntimeError("Index is empty.")
            self._matrix = torch.stack(self._vectors)
        return self._matrix

    def search(self, query: torch.Tensor, top_k: int = 5) -> list[list[RetrievalHit]]:
        """Return the top-k hits for each row of ``query`` ``(Q, dim)``."""
        if query.ndim == 1:
            query = query.unsqueeze(0)
        query = l2_normalize(query.detach().float())
        matrix = self._build()
        sims = query @ matrix.t()  # (Q, N)
        k = min(top_k, matrix.shape[0])
        scores, idx = sims.topk(k, dim=-1)
        results: list[list[RetrievalHit]] = []
        for row_scores, row_idx in zip(scores, idx):
            results.append(
                [
                    RetrievalHit(int(i), float(s), self._payloads[int(i)])
                    for s, i in zip(row_scores, row_idx)
                ]
            )
        return results

    def recall_at_k(self, queries: torch.Tensor, correct_indices: Sequence[int], k: int = 1) -> float:
        """Recall@k where ``correct_indices[q]`` is the gold item for query ``q``."""
        hits = self.search(queries, top_k=k)
        correct = 0
        for row, gold in zip(hits, correct_indices):
            if any(h.index == gold for h in row):
                correct += 1
        return correct / len(correct_indices)
