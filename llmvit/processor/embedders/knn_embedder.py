import torch
import torch.nn as nn
from functools import partial
from llmvit.processor.embedders._base_embedder import BaseEmbedder

class KNNEmbedder(BaseEmbedder):
    def __init__(self, embeddings: torch.Tensor, temperature: float = 1.0, k: int = 4, **kwargs):
        super().__init__(embeddings)
        from torch_cluster import knn
        import scipy.spatial
        
        assert temperature >= 0, "Temperature must be greater than 0"
        self.knn = knn
        self.temperature = temperature
        self.tree = scipy.spatial.cKDTree(embeddings.weight.data.cpu().numpy())
        self.k = k
        self.query = partial(
            self.tree.query, k=k, distance_upper_bound=embeddings.weight.data.size(1)
        )

    def _get_neighboor_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            src, tgt = self.knn(self.embeddings.weight.data, x.reshape(-1, x.shape[-1]), self.k)
            src_reshaped = src.reshape(x.shape[0], -1, self.k).permute(0, 2, 1)
            tgt_reshaped = tgt.reshape(x.shape[0], -1, self.k).permute(0, 2, 1)
            dist = torch.norm(self.embeddings(src_reshaped) - self.embeddings(tgt_reshaped), dim=-1)

        else:
            dist, tgt = self.query(x.detach().cpu().reshape(-1, x.shape[-1]))
            tgt_reshaped = tgt.reshape(x.shape[0], self.k, -1)
            emb_device = self.embeddings.weight.data.device
            tgt_reshaped = torch.from_numpy(tgt_reshaped).to(torch.long).to(emb_device)
            dist = dist.reshape(x.shape[0], self.k, -1)
            dist = torch.from_numpy(dist).to(x.dtype).to(emb_device)

        emb = self.embeddings(tgt_reshaped)
        return emb, dist

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb, dist = self._get_neighboor_embeddings(x)

        if self.k == 1:
            return {"inputs_embeds": emb.squeeze(1)}

        neighboor_weights = torch.softmax(
            -dist / (self.temperature + 1e-6), dim=1
        )
        inputs_embeds = torch.einsum("b k l, b k l d -> b l d", neighboor_weights, emb).to(x.device)
        return inputs_embeds


if __name__ == "__main__":
    num_embeddings = 100
    embedding_dim = 10
    embeddings = nn.Embedding(num_embeddings, embedding_dim)
    embedder = KNNEmbedder(embeddings, temperature=1.0, k=2)
    x = torch.randn(32, 8, embedding_dim)
    print(embedder(x))
