"""Decode a 32x32 quantized grid to RGB 64x64.

The default uses nearest-neighbor upsampling and convolutional refinement.
An optional zero-initialized detail head is retained for controlled experiments."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    embedding_dim: int = 128      # MUST match Encoder's embedding_dim / codebook dim
    hidden_channels: int = 256
    mid_channels: int = 128
    base_channels: int = 64
    out_channels: int = 3
    activation: str = "relu"
    detail_head: bool = False


def _act(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


def _upsample_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Nearest-neighbor 2x upsample followed by a stride-1 3x3 conv -
    an artifact-free way to upsample (see module docstring)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    )


class Decoder(nn.Module):
    """
    Input:  (N, embedding_dim, 32, 32)  - quantized latent grid
    Output: (N, 3, 64, 64), values in [-1, 1]
    """

    def __init__(self, cfg: DecoderConfig = DecoderConfig()):
        super().__init__()
        self.cfg = cfg
        act = cfg.activation

        self.net = nn.Sequential(
            # 32x32 -> 32x32, channel projection
            nn.Conv2d(cfg.embedding_dim, cfg.hidden_channels, kernel_size=1),
            nn.BatchNorm2d(cfg.hidden_channels),
            _act(act),

            # 32x32 -> 32x32, refine (mirrors the encoder's same-res convs)
            nn.Conv2d(cfg.hidden_channels, cfg.mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.mid_channels),
            _act(act),

            # 32x32 -> 64x64 (the only upsample)
            _upsample_conv(cfg.mid_channels, cfg.base_channels),
            nn.BatchNorm2d(cfg.base_channels),
            _act(act),

            # 64x64 -> 64x64, extra sharpening conv after upsampling
            nn.Conv2d(cfg.base_channels, cfg.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.base_channels),
            _act(act),

            # 64x64 -> 64x64, final RGB projection
            nn.Conv2d(cfg.base_channels, cfg.out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        if getattr(cfg, 'detail_head', False):
            # A learned 2x2 RGB correction directly from each quantized vector.
            # Zero initialization preserves the existing decoder exactly at
            # the start of a warm-started experiment. No raw-image skip exists.
            self.detail_head = nn.Sequential(
                nn.Conv2d(cfg.embedding_dim, cfg.out_channels * 4, 1, bias=False),
                nn.PixelShuffle(2),
            )
            nn.init.zeros_(self.detail_head[0].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, embedding_dim, 32, 32) - the quantized latent (Phase 5's output)
        if not hasattr(self, 'detail_head'):
            return self.net(z)
        x = z
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return torch.tanh(x + self.detail_head(z))


if __name__ == "__main__":
    from .encoder import Encoder, EncoderConfig

    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(embedding_dim=enc_cfg.embedding_dim)

    encoder = Encoder(enc_cfg)
    decoder = Decoder(dec_cfg)

    dummy_batch = torch.randn(8, 3, 64, 64)

    latent = encoder(dummy_batch)
    print("Encoder output (latent):", latent.shape)

    reconstruction = decoder(latent)
    print("Decoder output (reconstruction):", reconstruction.shape)

    assert reconstruction.shape == dummy_batch.shape, "Round-trip shape mismatch!"
    print("Round-trip shape matches input:", reconstruction.shape == dummy_batch.shape)
    print("Output value range:", reconstruction.min().item(), "to", reconstruction.max().item())

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {n_params:,}")

    loss = ((reconstruction - dummy_batch) ** 2).mean()
    loss.backward()
    enc_grad_ok = all(p.grad is not None for p in encoder.parameters())
    dec_grad_ok = all(p.grad is not None for p in decoder.parameters())
    print("Encoder gradients OK:", enc_grad_ok)
    print("Decoder gradients OK:", dec_grad_ok)

    print("Phase 4 decoder OK - full encoder-decoder round trip verified.")
