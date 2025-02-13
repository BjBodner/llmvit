import torch
import torch.nn as nn
from _base_embedder import BaseEmbedder

class MaxEmbedder(BaseEmbedder):
    def __init__(self, embeddings: torch.Tensor, similarity_metric: str = "cosine_similarity", **kwargs) -> None:
        """_summary_

        Args:
            embeddings (torch.Tensor): the embeddings layer of the pretrained LLM
            similarity_metric (str, optional): the similarity metric to use. options: ["cosine_similarity", "cdist"]. 
                                                Defaults to "cosine_similarity".
        """
        super().__init__(embeddings)
        self.similarity_metric = similarity_metric
        self.similarity_fn = getattr(torch.nn.functional, similarity_metric)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        max_idx = self.similarity_fn(x[:, :, None, :], self.embeddings.weight[None, None, :, :], dim=-1).argmax(-1)
        selected_embeddings = self.embeddings(max_idx)
        return selected_embeddings

if __name__ == "__main__":
    num_embeddings = 100
    embedding_dim = 10
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    embedder = MaxEmbedder(embedding_layer, temperature=1.0, k=2)
    x = torch.randn(32, 8, embedding_dim)
    print(embedder(x))
