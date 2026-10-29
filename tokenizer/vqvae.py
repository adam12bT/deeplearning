"""VQ-VAE model and discrete token-grid encode/decode helpers."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderConfig
from .decoder import Decoder, DecoderConfig
from .vector_quantizer import VectorQuantizer, VQConfig


@dataclass
class VQVAEConfig:
    in_channels: int = 3
    base_channels: int = 64
    mid_channels: int = 128
    hidden_channels: int = 256
    embedding_dim: int = 256    # doubled from 128 - each codebook entry is a
                                # longer vector, giving it more room to
                                # represent nuanced patch content (aimed at
                                # detail-heavy regions like city buildings)
    num_embeddings: int = 2048  # doubled from 1024 - a finer-grained "palette"
                                # of possible patches to choose from
    beta: float = 0.25
    activation: str = "relu"
    decoder_detail_head: bool = False


class VQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig = VQVAEConfig()):
        super().__init__()
        self.cfg = cfg

        enc_cfg = EncoderConfig(
            in_channels=cfg.in_channels, base_channels=cfg.base_channels,
            mid_channels=cfg.mid_channels, hidden_channels=cfg.hidden_channels,
            embedding_dim=cfg.embedding_dim, activation=cfg.activation,
        )
        vq_cfg = VQConfig(
            num_embeddings=cfg.num_embeddings, embedding_dim=cfg.embedding_dim,
            beta=cfg.beta,
        )
        dec_cfg = DecoderConfig(
            embedding_dim=cfg.embedding_dim, hidden_channels=cfg.hidden_channels,
            mid_channels=cfg.mid_channels, base_channels=cfg.base_channels,
            out_channels=cfg.in_channels, activation=cfg.activation,
            detail_head=getattr(cfg, 'decoder_detail_head', False),
        )

        self.encoder = Encoder(enc_cfg)
        self.vq = VectorQuantizer(vq_cfg)
        self.decoder = Decoder(dec_cfg)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        vq_out = self.vq(z)
        recon = self.decoder(vq_out["quantized"])

        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + vq_out["vq_loss"]

        return {
            "reconstruction": recon,
            "token_ids": vq_out["token_ids"],
            "recon_loss": recon_loss,
            "codebook_loss": vq_out["codebook_loss"],
            "commitment_loss": vq_out["commitment_loss"],
            "vq_loss": vq_out["vq_loss"],
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> token grid, no gradient tracking. Used in Phase 8 to
        convert the whole dataset into Transformer training sequences."""
        z = self.encoder(x)
        vq_out = self.vq(z)
        return vq_out["token_ids"]  # (N, 32, 32)

    @torch.no_grad()
    def decode_from_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token grid -> image. Used in Phase 21 to turn Transformer-generated
        tokens back into a viewable PNG."""
        quantized = self.vq.lookup(token_ids)
        return self.decoder(quantized)  # (N, 3, 64, 64), range [-1, 1]


if __name__ == "__main__":
    cfg = VQVAEConfig()
    model = VQVAE(cfg)

    dummy_batch = torch.randn(8, 3, 64, 64)
    out = model(dummy_batch)

    print("Reconstruction shape:", out["reconstruction"].shape)
    print("Token IDs shape:", out["token_ids"].shape)
    print("Total loss:", out["total_loss"].item())

    out["total_loss"].backward()
    all_grad_ok = all(p.grad is not None for p in model.parameters())
    print("All model parameters received gradients:", all_grad_ok)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total VQVAE parameters: {n_params:,}")

    # Test the two convenience methods
    token_ids = model.encode_to_indices(dummy_batch)
    print("encode_to_indices output shape:", token_ids.shape)
    recon_from_tokens = model.decode_from_indices(token_ids)
    print("decode_from_indices output shape:", recon_from_tokens.shape)

    print("Phase 6a VQVAE wrapper OK.")
