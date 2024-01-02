from typing import Optional
import numpy as np
import torch
from torchvision import datasets, transforms


class MNISTDataset(datasets.CIFAR10):

    N_CLASSES = 10
    def __init__(self, root: str, train: bool):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        super().__init__(root=root, train=train, download=True, transform=transform)
    def __getitem__(self, index):
        # x, y = self.data[index], self.targets[index]
        # x = self.transform(x.numpy())
        x, y = super().__getitem__(index)  # This directly uses the parent class method
        return x, y