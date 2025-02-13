import numpy as np
import torch
import pytest

from llmvit.utils.metrics import compute_metrics

def test_compute_metrics():
    # Create sample predictions and labels
    logits = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8]
    ])
    labels = torch.tensor([1, 0, 2])
    
    metrics = compute_metrics((logits, labels))
    
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert 0 <= metrics["accuracy"] <= 1

def test_perfect_accuracy():
    logits = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8]
    ])
    labels = torch.tensor([1, 0, 2])
    
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0

def test_zero_accuracy():
    logits = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1]
    ])
    labels = torch.tensor([1, 2, 0])
    
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 0.0 