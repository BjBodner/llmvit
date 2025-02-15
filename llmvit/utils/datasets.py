from typing import Dict

import torchvision

torchvision.disable_beta_transforms_warning()
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_dataset(name: str) -> Dict[str, Dataset]:
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip()
            if name != "MNIST"
            else transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset_constructor = getattr(datasets, name)
    train_dataset = dataset_constructor(
        root="./data", train=True, download=True, transform=train_transforms
    )
    eval_dataset = dataset_constructor(
        root="./data", train=False, download=True, transform=test_transforms
    )
    dataset = {"train": train_dataset, "eval": eval_dataset}
    return dataset
