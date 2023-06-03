import torch
import torch.nn as nn
import torchvision.models as models

from layers import (
    PatchExtractor, PatchEncoder, TransformerEncoder, MLP
)


def get_resnet_backbone(backbone_type=None):
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


class ConvClassifier(nn.Module):
    def __init__(self, resnet_type=None, num_classes=10):
        super(ConvClassifier, self).__init__()
        self._backbone = get_resnet_backbone(resnet_type)
        self._classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self._backbone(x)
        logits = self._classifier(features)
        return logits


class VitClassifier(nn.Module):
    def __init__(self, img_size=72, patch_size=6, embedding_dims=64, encoder_blocks=8, num_classes=10):
        super(VitClassifier, self).__init__()
        self._patch_ext = PatchExtractor(patch_size)
        patches_n = (img_size // patch_size) ** 2
        self._patch_encoder = PatchEncoder(
            patch_shape=(patches_n, 3, patch_size, patch_size),
            embedding_dims=embedding_dims,
        )
        self._norm = nn.LayerNorm(embedding_dims)
        self._do = nn.Dropout(p=0.5)
        self._encoders = nn.Sequential(
            *[
                TransformerEncoder(
                    embedding_dims=embedding_dims,
                    num_heads=4,
                ) for _ in range(encoder_blocks)
            ]
        )
        mlp_features = [
            patches_n * embedding_dims,
            2048,
            1024
        ]
        self._mlp = MLP(
            features=mlp_features,
            do=0.5
        )
        self._classifier = nn.Linear(
            in_features=mlp_features[-1],
            out_features=num_classes,
        )

    def forward(self, x):
        patches = self._patch_ext(x)
        embeddings = self._patch_encoder(patches)
        encoded_seq = self._encoders(embeddings)

        norm_seq = self._norm(encoded_seq)
        flatten_seq = torch.reshape(norm_seq, [norm_seq.shape[0], -1])
        flatten_seq = self._do(flatten_seq)

        features = self._mlp(flatten_seq)
        logits = self._classifier(features)
        return logits
