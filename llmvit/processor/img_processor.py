import torch
from torch import nn
from dataclasses import asdict

from llmvit.processor.img_encoders import EncoderConfig, IMG_ENCODER_REGISTRY
from llmvit.processor.embedders import EmbedderConfig, EMBEDDER_REGISTRY

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
        img_encoder_output = self.img_encoder(x)
        inputs_embeds = self.embedder(img_encoder_output)
        return inputs_embeds, img_encoder_output

