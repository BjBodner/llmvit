import torch
from ._base_embedder import BaseEmbedder


class DirectEmbedder(BaseEmbedder):

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return x
