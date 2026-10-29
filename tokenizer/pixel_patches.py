"""Discrete 2x2 RGB prototypes with direct, unsmoothed pixel reconstruction."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PixelPatchConfig:
    num_embeddings: int
    patch_size: int = 2
    in_channels: int = 3


class PixelPatchTokenizer(nn.Module):
    tokenizer_type = 'pixel_patch_v1'

    def __init__(self, codebook):
        super().__init__()
        if codebook.ndim != 2 or codebook.shape[1] != 12:
            raise ValueError('Expected an RGB 2x2 patch codebook of shape (K,12)')
        if not torch.isfinite(codebook).all() or codebook.min() < 0 or codebook.max() > 1:
            raise ValueError('Codebook pixels must be finite values in [0,1]')
        self.cfg = PixelPatchConfig(len(codebook))
        self.register_buffer('codebook', codebook.float().contiguous())

    @staticmethod
    def patches(images):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[2] % 2 or images.shape[3] % 2:
            raise ValueError('Expected RGB images with even spatial dimensions')
        grid = F.pixel_unshuffle((images.float() + 1) / 2, 2)
        return grid.permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def encode_to_indices(self, images):
        patches = self.patches(images)
        shape = patches.shape[:3]
        flat = patches.reshape(-1, 12)
        results = []
        norms = self.codebook.square().sum(1)
        for chunk in flat.split(2048):
            distance = chunk.square().sum(1, keepdim=True) - 2 * chunk @ self.codebook.T + norms
            results.append(distance.argmin(1))
        return torch.cat(results).reshape(shape)

    @torch.no_grad()
    def decode_from_indices(self, indices):
        if indices.ndim != 3:
            raise ValueError('Expected a batch of token grids')
        patches = F.embedding(indices, self.codebook).permute(0, 3, 1, 2).contiguous()
        return F.pixel_shuffle(patches, 2) * 2 - 1
