import torch
import torch.nn as nn
import math
import numpy as np


class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self._patch_size = patch_size
        self._unfold = nn.Unfold(
            kernel_size=self._patch_size,
            dilation=1,
            padding=0,
            stride=self._patch_size
        )

    def forward(self, x, as_images=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        assert isinstance(x, torch.Tensor)
        b, c, h, w = x.shape
        assert h % self._patch_size == 0 and w % self._patch_size == 0, \
            f'Image size is not multiple to pach size = {self._patch_size}'

        patches = self._unfold(x)
        patches = torch.permute(patches, dims=(0, 2, 1))
        if as_images:
            patches = torch.reshape(patches, (b, -1, c, self._patch_size, self._patch_size))
        return patches


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dims: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2) * (-math.log(10000.0) / embedding_dims))
        pe = torch.zeros(1, max_len, embedding_dims)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self._embedding_dims = embedding_dims
        self._max_len = max_len

    def forward(self, x):
        b, n, c = x.shape
        assert c == self._embedding_dims
        assert n <= self._max_len
        x = x + self.pe[0, :n]
        return self.dropout(x)


class PatchEncoder(nn.Module):
    def __init__(self, patch_shape, embedding_dims):
        super(PatchEncoder, self).__init__()
        n = patch_shape[0]
        features = torch.prod(torch.tensor(patch_shape[1:]))     

        self.patch_projection = nn.Linear(
            in_features=features,
            out_features=embedding_dims,
        )
        self.pos_encoding = PositionalEncoding(
            embedding_dims=embedding_dims,
            dropout=0.,
            max_len=n,
        )
        self._patch_shape = patch_shape

    def forward(self, x):
        b, n = x.shape[:2]
        assert x.shape[1:] == self._patch_shape, \
            f'Input shape: {x.shape}, patch shape: {self._patch_shape}'
        x = torch.reshape(x, (b, n, -1))
        embeddings = self.patch_projection(x)
        embeddings = self.pos_encoding(embeddings)
        return embeddings


class MLP(nn.Module):
    def __init__(self, features, act=nn.GELU, do=0.1):
        super(MLP, self).__init__()
        assert isinstance(features, list)
        assert len(features) >= 2

        out_features = features[1:]
        in_features, layers = features[0], []
        for out_feat in out_features:
            layers.extend(
                [
                    nn.Linear(
                        in_features=in_features,
                        out_features=out_feat,
                    ),
                    act(),
                    nn.Dropout(do)
                ]
            )
            in_features = out_feat

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dims, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self._norm = nn.LayerNorm(embedding_dims)
        self._attention = torch.nn.MultiheadAttention(
            embed_dim=embedding_dims,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
        )
        self._mlp = MLP(
            features=[embedding_dims, 2 * embedding_dims, embedding_dims]
        )

    def forward(self, x, return_scores=False):
        in_sequence = self._norm(x)
        attention, scores = self._attention(
            query=in_sequence,
            key=in_sequence,
            value=in_sequence,
        )
        attention = attention + in_sequence

        mlp_in_sequence = self._norm(attention)
        out_features = self._mlp(mlp_in_sequence)
        out_features = out_features + attention

        if return_scores:
            return out_features, scores
        else:
            return out_features

