import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from typing import Any, Dict

import model


def build_core(core_type, *args, **kwargs):
    model_cls = getattr(model, core_type)
    return model_cls(
        *args,
        **kwargs,
    )


def compute_accuracy(logits, targets):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy


class ImageClassifier(pl.LightningModule):
    def __init__(self, core_type='ConvClassifier', num_classes=10, *args, **kwargs):
        super().__init__()
        self._core = build_core(core_type, num_classes=num_classes)
        self._criterion = nn.CrossEntropyLoss()

    @staticmethod
    def preprocess(images_batch, resize=(72, 72)):
        assert len(images_batch.shape) == 4
        if isinstance(images_batch, np.ndarray):
            images_batch = torch.from_numpy(images_batch)
        assert torch.is_tensor(images_batch)
        if images_batch.shape[1] != 3 and images_batch.shape[3] == 3:
            images_batch = torch.permute(images_batch, (0, 3, 1, 2))
        images_batch = images_batch.float() / 255.
        images_batch = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(images_batch)
        if resize is not None:
            images_batch = torchvision.transforms.Resize(resize)(images_batch)
        return images_batch

    def forward(self, images_batch):
        images_batch = ImageClassifier.preprocess(images_batch)
        logits = self._core(images_batch)

        if not self.training:
            scores = torch.softmax(logits, dim=-1)
            max_scores, max_indices = torch.max(scores, dim=1)
            return max_scores, max_indices
        else:
            return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [[optimizer], [scheduler]]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self.forward(images)
        loss = self._criterion(logits, labels)
        acc = compute_accuracy(logits, labels)
        self.log_dict(
            {
                f"loss/train": loss.detach(),
                f"accuracy/train": acc,
            }
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        if batch_idx == 0:
            images, labels = batch
            _, classes = self.forward(images)
            accuracy = (classes == labels).float().mean()
            self.log_dict(
                {
                    f"accuracy/val": accuracy,
                }
            )
