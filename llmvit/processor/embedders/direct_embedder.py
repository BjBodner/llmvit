import torch
from llmvit.processor.embedders._base_embedder import BaseEmbedder


class DirectEmbedder(BaseEmbedder):

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return x
