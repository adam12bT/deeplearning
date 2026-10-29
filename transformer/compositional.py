"""Autoregressive visual tokens conditioned on separate named feature tokens."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from transformer.blocks import Block


@dataclass
class CompositionalConfig:
    vocab_size: int
    block_size: int
    condition_sizes: tuple
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = .1

    def __post_init__(self):
        self.condition_sizes=tuple(self.condition_sizes)
        if not self.condition_sizes or min(self.condition_sizes)<1:
            raise ValueError('Nonempty condition vocabularies are required')
        if self.n_embd%self.n_head: raise ValueError('Embedding width must be divisible by heads')
        if self.block_size<1 or self.vocab_size<1: raise ValueError('Invalid visual vocabulary/grid')


class CompositionalTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.token_emb=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        # Separate embeddings preserve each feature's identity instead of
        # collapsing all five controls into one summed starting vector.
        self.condition_emb=nn.ModuleList([nn.Embedding(size,cfg.n_embd) for size in cfg.condition_sizes])
        self.prefix_size=len(cfg.condition_sizes)
        self.pos_emb=nn.Parameter(torch.empty(1,cfg.block_size+self.prefix_size-1,cfg.n_embd))
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f=nn.LayerNorm(cfg.n_embd)
        self.lm_head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        nn.init.trunc_normal_(self.pos_emb,std=.02)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.trunc_normal_(module.weight,std=.02)
            if isinstance(module,nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)

    def _hidden(self,previous_tokens,conditions):
        if conditions.shape!=(previous_tokens.shape[0],self.prefix_size):
            raise ValueError(f'Expected conditions with shape (batch,{self.prefix_size})')
        if previous_tokens.shape[1]>=self.cfg.block_size:
            raise ValueError('Too many previous image tokens')
        prefix=torch.stack([embedding(conditions[:,i]) for i,embedding in enumerate(self.condition_emb)],dim=1)
        x=torch.cat([prefix,self.token_emb(previous_tokens)],dim=1)
        x=self.drop(x+self.pos_emb[:,:x.shape[1]])
        for block in self.blocks: x=block(x)
        return self.ln_f(x)

    def forward(self,image_tokens,conditions,compute_loss=True):
        if image_tokens.ndim!=2 or image_tokens.shape[1]!=self.cfg.block_size:
            raise ValueError(f'Expected {self.cfg.block_size} image tokens')
        hidden=self._hidden(image_tokens[:,:-1],conditions)
        # The last feature token predicts image token 0. Prefix positions
        # before it have no image targets and do not contribute to loss.
        logits=self.lm_head(hidden[:,self.prefix_size-1:])
        loss=F.cross_entropy(logits.reshape(-1,self.cfg.vocab_size),image_tokens.reshape(-1)) if compute_loss else None
        return logits,loss

    @torch.no_grad()
    def generate(self,conditions,temperature=1.,top_k=50):
        if temperature<=0: raise ValueError('temperature must be positive')
        if top_k is not None and top_k<1: raise ValueError('top_k must be positive or None')
        self.eval()
        generated=torch.empty((len(conditions),0),dtype=torch.long,device=conditions.device)
        for _ in range(self.cfg.block_size):
            logits=self.lm_head(self._hidden(generated,conditions)[:,-1])/temperature
            if top_k is not None:
                cutoff=torch.topk(logits,min(top_k,self.cfg.vocab_size)).values[:,-1:]
                logits=logits.masked_fill(logits<cutoff,-float('inf'))
            token=torch.multinomial(F.softmax(logits,dim=-1),1)
            generated=torch.cat([generated,token],dim=1)
        return generated
