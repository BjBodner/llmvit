import torch
import pytest

from llmvit.processor.img_encoders import ViTEncoder, PatchEmbed

@pytest.fixture
def sample_image() -> torch.Tensor:
    batch_size = 2
    channels = 3
    img_size = 224
    return torch.randn(batch_size, channels, img_size, img_size)

def test_patch_embed() -> None:
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    
    encoder = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim
    )
    
    x = torch.randn(2, in_channels, img_size, img_size)
    output = encoder(x)
    
    expected_seq_len = (img_size // patch_size) ** 2
    assert output.shape == (2, expected_seq_len, embed_dim)

def test_vit_encoder() -> None:
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    
    encoder = ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_encoder_layers=2
    )
    
    x = torch.randn(2, in_channels, img_size, img_size)
    output = encoder(x)
    
    expected_seq_len = (img_size // patch_size) ** 2
    assert output.shape == (2, expected_seq_len, embed_dim)

def test_invalid_patch_size() -> None:
    with pytest.raises(AssertionError):
        PatchEmbed(img_size=224, patch_size=15)  # Patch size should divide image size 