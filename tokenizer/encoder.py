"""Convolutional encoder: RGB 64x64 to a 32x32 latent grid.

The grid spacing is two pixels; each latent has a larger convolutional receptive field."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    in_channels: int = 3
    base_channels: int = 64       # channels after the one downsample
    mid_channels: int = 128       # channels in the first same-resolution refine conv
    hidden_channels: int = 256    # channels in the second same-resolution refine conv
    embedding_dim: int = 128      # MUST match VectorQuantizer's embedding_dim
    activation: str = "relu"      # "relu" or "gelu"


def _act(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


class Encoder(nn.Module):
    """
    Input:  (N, in_channels, 64, 64)
    Output: (N, embedding_dim, 32, 32)
    """

    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.cfg = cfg
        act = cfg.activation

        self.net = nn.Sequential(
            # 64x64 -> 32x32 (the only spatial downsample)
            nn.Conv2d(cfg.in_channels, cfg.base_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cfg.base_channels),
            _act(act),

            # 32x32 -> 32x32 (same resolution, adds representational capacity)
            nn.Conv2d(cfg.base_channels, cfg.mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cfg.mid_channels),
            _act(act),

            # 32x32 -> 32x32 (same resolution, adds more capacity)
            nn.Conv2d(cfg.mid_channels, cfg.hidden_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cfg.hidden_channels),
            _act(act),

            # 32x32 -> 32x32, channel projection only (no spatial mixing)
            nn.Conv2d(cfg.hidden_channels, cfg.embedding_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 64, 64), values in [-1, 1] (matches Phase 2's normalization)
        return self.net(x)  # -> (N, embedding_dim, 32, 32)


if __name__ == "__main__":
    # Sanity check: does the shape math actually work out as described above?
    cfg = EncoderConfig()
    encoder = Encoder(cfg)

    dummy_batch = torch.randn(8, 3, 64, 64)  # pretend batch, like Phase 2's output
    out = encoder(dummy_batch)

    print("Input shape: ", dummy_batch.shape)
    print("Output shape:", out.shape)
    assert out.shape == (8, cfg.embedding_dim, 32, 32), "Shape mismatch!"

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {n_params:,}")

    # Gradient check: does backprop actually flow through every layer?
    loss = out.mean()
    loss.backward()
    grad_ok = all(p.grad is not None for p in encoder.parameters())
    print("All parameters received gradients:", grad_ok)

    print("Phase 3 encoder OK.")