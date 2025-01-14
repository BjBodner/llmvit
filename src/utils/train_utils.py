import torch

def image_collator(features: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    images = torch.stack([f[0] for f in features])
    labels = torch.tensor([f[1] for f in features])
    return {"images": images, "labels": labels}