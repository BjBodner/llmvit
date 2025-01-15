import torch
from torch import nn

from llmvit.nn.encoders.patch_embed import PatchEmbed


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 3,
        hidden_dim: int = 512,
        embed_dim: int = 768,
        num_encoder_layers: int = 4,
        dropout: float = 0.1,
        nhead: int = 12,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                nhead=nhead,
            ),
            num_layers=num_encoder_layers,
        )
        self.output_layer = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        x = self.norm(x)
        return x
