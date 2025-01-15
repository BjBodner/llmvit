import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size: int = 28, patch_size: int = 7, embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.LazyConv2d(embed_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        x += self.pos_embed
        x = self.norm(x)
        return x
