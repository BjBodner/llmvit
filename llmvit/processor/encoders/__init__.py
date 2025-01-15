import torch.nn as nn
from .vit_encoder import ViTEncoder
from .patch_embed import PatchEmbed

ENCODER_REGISTRY = {"vit": ViTEncoder, "patch_embed": PatchEmbed}


__all__ = ["ViTEncoder", "PatchEmbed"]
