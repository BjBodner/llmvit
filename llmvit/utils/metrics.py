import numpy as np
import torch
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}
