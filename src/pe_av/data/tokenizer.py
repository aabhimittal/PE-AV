"""A tiny, dependency-free word tokenizer.

Real PE-AV uses a large BPE vocabulary; for a self-contained, reproducible demo
we use a deterministic hashing tokenizer.  Words are lower-cased, split on
non-alphanumeric characters, and hashed into a fixed vocabulary.  This needs no
vocab file and round-trips ids stably across runs, which is all the training /
retrieval demos require.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_WORD_RE = re.compile(r"[a-z0-9]+")

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NUM_SPECIAL = 3


@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor  # (B, L)
    attention_mask: torch.Tensor  # (B, L), 1 for real tokens


class HashTokenizer:
    def __init__(self, vocab_size: int = 32000, max_length: int = 32):
        if vocab_size <= NUM_SPECIAL:
            raise ValueError("vocab_size must be larger than the number of special tokens.")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _hash(self, word: str) -> int:
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        return NUM_SPECIAL + (int(digest, 16) % (self.vocab_size - NUM_SPECIAL))

    def encode(self, text: str) -> list[int]:
        words = _WORD_RE.findall(text.lower())
        ids = [BOS_ID] + [self._hash(w) for w in words]
        ids = ids[: self.max_length - 1] + [EOS_ID]
        return ids[: self.max_length]

    def __call__(self, texts: Sequence[str] | str) -> TokenizerOutput:
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        length = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), length), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), length, dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return TokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)
