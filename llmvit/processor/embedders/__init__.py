from torch import nn
from .knn_embedder import KNNEmbedder
from .direct_embedder import DirectEmbedder

EMBEDDER_REGISTRY = {"knn": KNNEmbedder, "direct": DirectEmbedder}

__all__ = ["KNNEmbedder", "DirectEmbedder"]
