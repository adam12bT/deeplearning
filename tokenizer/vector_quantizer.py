"""Nearest-neighbor visual codebook with straight-through encoder gradients.

Vocabulary size and embedding width are configured by the VQ-VAE."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQConfig:
    num_embeddings: int = 512   # codebook size (vocabulary of visual tokens)
    embedding_dim: int = 128    # MUST match Encoder/Decoder embedding_dim
    beta: float = 0.25          # commitment loss weight


class VectorQuantizer(nn.Module):
    def __init__(self, cfg: VQConfig = VQConfig()):
        super().__init__()
        self.cfg = cfg

        # The codebook itself: 512 learnable 128-dim vectors.
        # nn.Embedding is just a lookup table (num_embeddings x embedding_dim)
        # that also happens to support gradient updates - exactly what we need.
        self.codebook = nn.Embedding(cfg.num_embeddings, cfg.embedding_dim)
        # Initialize uniformly in a small range, standard VQ-VAE practice -
        # keeps initial distances reasonable so early training isn't unstable.
        self.codebook.weight.data.uniform_(
            -1.0 / cfg.num_embeddings, 1.0 / cfg.num_embeddings
        )

    def forward(self, z: torch.Tensor):
        # z: (N, C, H, W) with C == embedding_dim
        N, C, H, W = z.shape
        assert C == self.cfg.embedding_dim, \
            f"Encoder output channels ({C}) must match VQ embedding_dim ({self.cfg.embedding_dim})"

        # Rearrange so each spatial position becomes one row: (N*H*W, C)
        # This is the "flatten into 256 separate 128-dim vectors" step.
        z_perm = z.permute(0, 2, 3, 1).contiguous()   # (N, H, W, C)
        z_flat = z_perm.view(-1, C)                     # (N*H*W, C)

        # --- Distance computation: ||z - e||^2 = ||z||^2 - 2 z.e^T + ||e||^2 ---
        codebook_weight = self.codebook.weight            # (num_embeddings, C)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)      # (N*H*W, 1)
        e_sq = (codebook_weight ** 2).sum(dim=1)           # (num_embeddings,)
        cross_term = z_flat @ codebook_weight.t()          # (N*H*W, num_embeddings)
        distances = z_sq - 2 * cross_term + e_sq           # (N*H*W, num_embeddings)

        # --- Nearest neighbor lookup ---
        token_ids_flat = torch.argmin(distances, dim=1)     # (N*H*W,)
        quantized_flat = self.codebook(token_ids_flat)      # (N*H*W, C)

        quantized = quantized_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        token_ids = token_ids_flat.view(N, H, W)             # (N, H, W)

        # --- The two VQ-specific losses ---
        # codebook_loss: pulls codebook vectors toward encoder output (encoder side detached)
        codebook_loss = F.mse_loss(quantized, z.detach())
        # commitment_loss: pulls encoder output toward the codebook vector it picked (codebook side detached)
        commitment_loss = F.mse_loss(z, quantized.detach())

        # --- Straight-through estimator ---
        # Forward value = quantized (real, discrete). Backward gradient =
        # as if this were the identity function on z (see docstring above).
        quantized_st = z + (quantized - z).detach()

        return {
            "quantized": quantized_st,           # (N, C, H, W) - feed this to the Decoder
            "token_ids": token_ids,              # (N, H, W) - the visual token grid
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": codebook_loss + self.cfg.beta * commitment_loss,
        }

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Given token IDs (e.g. from the Transformer's generation, Phase 20+),
        looks up their embeddings and reshapes to (N, C, H, W) for the Decoder.
        token_ids: (N, H, W) integers."""
        N, H, W = token_ids.shape
        flat = token_ids.view(-1)
        emb = self.codebook(flat)                              # (N*H*W, C)
        return emb.view(N, H, W, self.cfg.embedding_dim).permute(0, 3, 1, 2).contiguous()


if __name__ == "__main__":
    from .encoder import Encoder, EncoderConfig
    from .decoder import Decoder, DecoderConfig

    enc_cfg = EncoderConfig()
    vq_cfg = VQConfig(embedding_dim=enc_cfg.embedding_dim)
    dec_cfg = DecoderConfig(embedding_dim=enc_cfg.embedding_dim)

    encoder = Encoder(enc_cfg)
    vq = VectorQuantizer(vq_cfg)
    decoder = Decoder(dec_cfg)

    dummy_batch = torch.randn(8, 3, 64, 64)

    z = encoder(dummy_batch)
    print("Encoder output z:", z.shape)

    vq_out = vq(z)
    print("Quantized shape:", vq_out["quantized"].shape)
    print("Token IDs shape:", vq_out["token_ids"].shape)
    print("Token ID range: ", vq_out["token_ids"].min().item(), "to", vq_out["token_ids"].max().item())
    print("Codebook loss:  ", vq_out["codebook_loss"].item())
    print("Commitment loss:", vq_out["commitment_loss"].item())

    recon = decoder(vq_out["quantized"])
    print("Reconstruction shape:", recon.shape)
    assert recon.shape == dummy_batch.shape

    # Full loss, like Phase 6's training loop will compute
    recon_loss = F.mse_loss(recon, dummy_batch)
    total_loss = recon_loss + vq_out["vq_loss"]
    total_loss.backward()

    enc_grad_ok = all(p.grad is not None for p in encoder.parameters())
    dec_grad_ok = all(p.grad is not None for p in decoder.parameters())
    codebook_grad_ok = vq.codebook.weight.grad is not None

    print("Encoder gradients OK (via straight-through):", enc_grad_ok)
    print("Decoder gradients OK:", dec_grad_ok)
    print("Codebook gradients OK:", codebook_grad_ok)

    # Test the reverse lookup path too (used later by the Transformer's generation)
    reconstructed_z = vq.lookup(vq_out["token_ids"])
    print("Lookup round-trip shape:", reconstructed_z.shape)
    assert torch.allclose(reconstructed_z, vq_out["quantized"].detach(), atol=1e-5)
    print("Token-ID -> embedding lookup matches original quantized values.")

    print("Phase 5 vector quantizer OK - full encoder-VQ-decoder pipeline verified.")
