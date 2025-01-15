import torch
from torch import nn
from typing import Dict, Any

from llmvit.processor.encoders import ENCODER_REGISTRY
from llmvit.processor.embedders import EMBEDDER_REGISTRY

DEFAULT_IMG_ENCODER_ARGS = {"type": "vit"}
DEFAULT_EMBEDDER_ARGS = {"type": "knn"}


class ImgProcessor(nn.Module):
    def __init__(
        self,
        text_embeddings: nn.Module,
        img_encoder_args: Dict[str, Any] = DEFAULT_IMG_ENCODER_ARGS,
        embedder_args: Dict[str, Any] = DEFAULT_EMBEDDER_ARGS,
    ) -> None:
        super().__init__()
        self.img_encoder = ENCODER_REGISTRY(img_encoder_args["type"])(
            **img_encoder_args
        )
        self.embedder = EMBEDDER_REGISTRY(embedder_args["type"])(
            text_embeddings, **embedder_args
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.img_encoder(x)
        out_dict = self.embedder(x)
        return out_dict


# issues:
# 2. we want to provide a way to abstract away the image part, to give the user freedom - I can make a default
