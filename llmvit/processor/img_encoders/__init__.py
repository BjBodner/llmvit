from dataclasses import dataclass
from .vit_encoder import ViTEncoder
from .patch_embed import PatchEmbed

@dataclass
class EncoderConfig:
    type: str = "vit"
    img_size: int = 28
    patch_size: int = 7
    in_channels: int = 3
    hidden_dim: int = 256
    embed_dim: int = 768
    num_encoder_layers: int = 2
    dropout: float = 0.0
    nhead: int = 8

IMG_ENCODER_REGISTRY = {"vit": ViTEncoder, "patch_embed": PatchEmbed}


__all__ = ["ViTEncoder", "PatchEmbed"]
