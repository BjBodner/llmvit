import torch
import torch.nn as nn
import pytest

from llmvit.processor.embedders import KNNEmbedder, MaxEmbedder, DirectEmbedder

@pytest.fixture
def sample_embeddings() -> nn.Embedding:
    num_embeddings = 100
    embedding_dim = 768
    return nn.Embedding(num_embeddings, embedding_dim)

@pytest.fixture
def sample_input() -> torch.Tensor:
    batch_size = 4
    seq_len = 16
    embedding_dim = 768
    return torch.randn(batch_size, seq_len, embedding_dim)

def test_knn_embedder(sample_embeddings: nn.Embedding, sample_input: torch.Tensor) -> None:
    embedder = KNNEmbedder(sample_embeddings, temperature=1.0, k=4)
    output = embedder(sample_input)
    assert isinstance(output, dict)
    assert "inputs_embeds" in output
    assert output["inputs_embeds"].shape == sample_input.shape

def test_max_embedder(sample_embeddings: nn.Embedding, sample_input: torch.Tensor) -> None:
    embedder = MaxEmbedder(sample_embeddings)
    output = embedder(sample_input)
    assert isinstance(output, dict)
    assert "inputs_embeds" in output
    assert output["inputs_embeds"].shape == sample_input.shape

def test_direct_embedder(sample_embeddings: nn.Embedding, sample_input: torch.Tensor) -> None:
    embedder = DirectEmbedder(sample_embeddings)
    output = embedder(sample_input)
    assert output.shape == sample_input.shape

def test_invalid_temperature() -> None:
    with pytest.raises(AssertionError):
        KNNEmbedder(sample_embeddings(), temperature=-1.0) 