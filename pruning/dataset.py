import numpy as np
import pytorch_lightning as pl
from typing import Any
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class ToTensor:
    def __call__(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr
        arr = np.asarray(arr)
        return torch.from_numpy(arr)


class CIFAR10(pl.LightningDataModule):
    def __init__(self, train_batch=64, val_batch=32):
        super().__init__()
        self._train_batch = train_batch
        self._val_batch = val_batch
        self._train_xform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            ToTensor(),
        ])
        self._val_xform = transforms.Compose([
            ToTensor(),
        ])

    def train_dataloader(self) -> DataLoader:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self._train_xform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self._train_batch,
                                                shuffle=True, num_workers=2)
        return trainloader

    def val_dataloader(self) -> DataLoader:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self._val_xform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self._val_batch,
                                                shuffle=False, num_workers=2)
        return testloader
