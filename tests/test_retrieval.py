import pytest
import torch

from pe_av.retrieval import EmbeddingIndex


def test_add_and_search_exact_match():
    idx = EmbeddingIndex(dim=4)
    vecs = torch.eye(4)
    idx.add(vecs, ["a", "b", "c", "d"])
    assert len(idx) == 4
    hits = idx.search(torch.tensor([[1.0, 0, 0, 0]]), top_k=2)[0]
    assert hits[0].payload == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_dim_mismatch_raises():
    idx = EmbeddingIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add(torch.randn(2, 8), ["x", "y"])


def test_payload_count_mismatch_raises():
    idx = EmbeddingIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add(torch.randn(2, 4), ["only-one"])


def test_recall_at_k():
    idx = EmbeddingIndex(dim=3)
    idx.add(torch.eye(3), [0, 1, 2])
    queries = torch.eye(3)  # each query matches its own row
    assert idx.recall_at_k(queries, [0, 1, 2], k=1) == 1.0


def test_topk_larger_than_index():
    idx = EmbeddingIndex(dim=2)
    idx.add(torch.eye(2), ["a", "b"])
    hits = idx.search(torch.tensor([[1.0, 0.0]]), top_k=10)[0]
    assert len(hits) == 2
