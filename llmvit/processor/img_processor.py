import torch
from torch import nn
from typing import Dict, Any
from dataclasses import asdict

from processor.img_encoders import EncoderConfig, IMG_ENCODER_REGISTRY
from processor.embedders import EmbedderConfig, EMBEDDER_REGISTRY

DEFAULT_IMG_ENCODER_RGSA = {"type": "vit"}

class ImgProcessor(nn.Module):
    def __init__(
        self,
        word_embeddings: nn.Module,
        img_encoder_config: EncoderConfig = EncoderConfig,
        embedder_config: EmbedderConfig = EmbedderConfig,
    ) -> None:
        super().__init__()
        self.img_encoder = IMG_ENCODER_REGISTRY[img_encoder_config.type](
            **asdict(img_encoder_config)
        )
        self.embedder = EMBEDDER_REGISTRY[embedder_config.type](
            word_embeddings, **asdict(embedder_config)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.img_encoder(x)
        out_dict = self.embedder(x)
        return out_dict

