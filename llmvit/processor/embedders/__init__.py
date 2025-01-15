from torch import nn
from dataclasses import dataclass
from .knn_embedder import KNNEmbedder
from .direct_embedder import DirectEmbedder

EMBEDDER_REGISTRY = {"knn": KNNEmbedder, "direct": DirectEmbedder}


@dataclass
class EmbedderConfig:
    type: str = "knn"
    k: int = 4
    temperature: float = 1.0


__all__ = ["KNNEmbedder", "DirectEmbedder"]
