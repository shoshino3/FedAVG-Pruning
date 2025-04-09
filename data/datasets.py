from typing import Optional
import numpy as np
import torch
from torchvision import datasets, transforms


class MNISTDataset(datasets.MNIST):

    N_CLASSES = 10

    def __init__(self, root: str, train: bool):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        super().__init__(root=root, train=train, download=True, transform=transform)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)

        return x, y

class CIFAR10Dataset(datasets.CIFAR10):

    N_CLASSES = 10
    
    def __init__(self, root: str, train: bool, data_augmentation: bool = False):
        if data_augmentation:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        else:
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
    
class CIFAR100Dataset(datasets.CIFAR100):
    
    N_CLASSES = 100

    def __init__(self, root: str, train: bool, data_augmentation: bool = False):
        if data_augmentation:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4865, 0.4409), 
                        (0.2673, 0.2564, 0.2762), 
                    ),
                ]
            )
        super().__init__(root=root, train=train, download=True, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y