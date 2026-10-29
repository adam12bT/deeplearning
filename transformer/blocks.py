"""Reusable causal attention and feed-forward Transformer blocks."""
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position i can only attend to
    positions <= i (enforced via `is_causal=True` in scaled_dot_product_attention,
    which uses a fused/flash-attention kernel on CUDA when available)."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C = x.shape

        qkv = self.qkv_proj(x)                              # (N, T, 3C)
        q, k, v = qkv.split(C, dim=2)                        # each (N, T, C)

        # Reshape to (N, n_head, T, head_dim) for per-head attention.
        q = q.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(N, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fused scaled-dot-product attention with causal masking built in.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )                                                    # (N, n_head, T, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(N, T, C)
        return self.resid_dropout(self.out_proj(attn_out))


class MLP(nn.Module):
    """Standard Transformer feed-forward block: expand 4x, GELU, project back."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block: LN -> attn -> residual, LN -> MLP -> residual.
    Pre-norm (norm before the sublayer, not after) is the standard modern
    choice - it makes deep Transformers noticeably easier to train."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
