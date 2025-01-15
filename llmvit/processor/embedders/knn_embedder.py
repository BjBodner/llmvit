import torch
import torch.nn as nn
import scipy.spatial
from _base_embedder import BaseEmbedder
from functools import partial


class KNNEmbedder(BaseEmbedder):
    def __init__(self, embeddings: torch.Tensor, temperature: float = 1.0, k: int = 4):
        super().__init__(embeddings)
        assert temperature >= 0, "Temperature must be greater than 0"
        self.temperature = temperature
        self.tree = scipy.spatial.cKDTree(embeddings.weight.data.numpy())
        self.k = k
        self.query = partial(
            self.tree.query, k=k, distance_upper_bound=embeddings.weight.data.size(1)
        )

    def _get_neighboor_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        dist, col = self.query(x.detach().cpu().reshape(-1, x.shape[-1]))
        col = col.reshape(x.shape[0], self.k, -1)
        dist = dist.reshape(x.shape[0], self.k, -1)
        emb = self.embeddings(
            torch.from_numpy(col).to(torch.long).to(self.embeddings.weight.data.device)
        )
        return emb, dist

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb, dist = self._get_neighboor_embeddings(x)

        if self.k == 1:
            return {"inputs_embeds": emb.squeeze(1)}

        neighboor_weights = torch.softmax(
            -torch.from_numpy(dist).to(x.dtype) / (self.temperature + 1e-6), dim=1
        )
        inputs_embeds = torch.einsum("b k l, b k l d -> b l d", neighboor_weights, emb)
        return {"inputs_embeds": inputs_embeds}


if __name__ == "__main__":
    num_embeddings = 100
    embedding_dim = 10
    embeddings = nn.Embedding(num_embeddings, embedding_dim)
    embedder = KNNEmbedder(embeddings, temperature=1.0, k=2)
    x = torch.randn(32, 8, embedding_dim)
    print(embedder(x))
