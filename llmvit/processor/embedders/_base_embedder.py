import torch
from torch import nn


class BaseEmbedder(nn.Module):
    def __init__(self, embeddings: torch.Tensor = None, **kwargs):
        super().__init__()
        self.embeddings = embeddings
