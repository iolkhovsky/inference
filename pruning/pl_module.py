import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from typing import Any, Dict


def get_resnet_backbone(backbone_type=None, pretrained=True):
    if backbone_type is None:
        backbone_type = 18
    assert backbone_type in [18, 34, 50, 101, 152], f'Unsupported backbone: {backbone_type}'
    model_cls = getattr(models, f'resnet{backbone_type}')
    weights = getattr(models, f'ResNet{backbone_type}_Weights')
    model = model_cls(weights=weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(
        *list(list(model.children())[:-1])
    )
    return backbone


def compute_accuracy(logits, targets):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy


class ResnetClassifier(pl.LightningModule):
    def __init__(self, resnet_type=None, num_classes=10, *args, **kwargs):
        super().__init__()
        self._backbone = get_resnet_backbone(resnet_type)
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
        self._criterion = nn.CrossEntropyLoss()

    @staticmethod
    def preprocess(images_batch):
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
        return images_batch


    def forward(self, images_batch):
        images_batch = ResnetClassifier.preprocess(images_batch)
        features = self._backbone(images_batch)
        logits = self._classifier(features)

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
