from torch import nn
from dataclasses import dataclass
from .knn_embedder import KNNEmbedder
from .max_embedder import MaxEmbedder
from .direct_embedder import DirectEmbedder

EMBEDDER_REGISTRY = {"knn": KNNEmbedder, "direct": DirectEmbedder, "max": MaxEmbedder}


@dataclass
class EmbedderConfig:
    type: str = "knn"
    k: int = 4 # only used for knn
    temperature: float = 0.1 # only used for knn
    similarity_metric: str = "cosine_similarity" # only used for max

__all__ = ["KNNEmbedder", "DirectEmbedder", "MaxEmbedder"]
