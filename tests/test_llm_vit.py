import torch
import pytest
from transformers import AutoModelForSequenceClassification

from llmvit import LLMVIT, EncoderConfig, EmbedderConfig

@pytest.fixture
def sample_model() -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=10,
        num_hidden_layers=2
    )

@pytest.fixture
def sample_images() -> torch.Tensor:
    return torch.randn(2, 3, 224, 224)

@pytest.fixture
def sample_labels() -> torch.Tensor:
    return torch.tensor([0, 1])

def test_llm_vit_forward(
    sample_model: AutoModelForSequenceClassification, 
    sample_images: torch.Tensor, 
    sample_labels: torch.Tensor
) -> None:
    model = LLMVIT(
        model=sample_model,
        img_processor_config=EncoderConfig(
            type="vit",
            img_size=224,
            patch_size=16
        ),
        embedder_config=EmbedderConfig(
            type="knn",
            k=4
        )
    )
    
    outputs = model(sample_images, sample_labels)
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (2, 10)

def test_backbone_freezing(sample_model: AutoModelForSequenceClassification) -> None:
    model = LLMVIT(
        model=sample_model,
        frozen_backbone_steps=10
    )
    
    # Check initial frozen state
    for param in model.model.parameters():
        if "classifier" not in param:
            assert not param.requires_grad
            
    # Check unfreezing after steps
    for _ in range(11):
        model.increment_frozen_backbone_steps()
        
    for param in model.model.parameters():
        assert param.requires_grad

def test_embedding_loss(
    sample_model: AutoModelForSequenceClassification, 
    sample_images: torch.Tensor, 
    sample_labels: torch.Tensor
) -> None:
    model = LLMVIT(
        model=sample_model,
        embedding_loss_weight=0.1
    )
    
    outputs = model(sample_images, sample_labels)
    assert outputs["loss"] is not None 