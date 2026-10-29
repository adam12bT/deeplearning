"""Load the project's VQ-VAE checkpoints, including older module aliases."""
import sys
from pathlib import Path

import torch

from . import vqvae as vqvae_module
from .vqvae import VQVAE, VQVAEConfig

# Early checkpoints stored their config under the bare module name.
sys.modules.setdefault('vqvae', vqvae_module)


def load_vqvae(checkpoint_path: Path, device: torch.device) -> VQVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get('model_type') == 'pixel_patch_v1':
        raise ValueError('This is a pixel-patch tokenizer, not a VQ-VAE; use load_tokenizer')
    return _vqvae_from_checkpoint(checkpoint, device, checkpoint_path)


def _vqvae_from_checkpoint(checkpoint, device, checkpoint_path):
    config = checkpoint.get('config', VQVAEConfig())
    if isinstance(config, dict):
        config = VQVAEConfig(**config)
    model = VQVAE(config).to(device)
    state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
    if state is None:
        raise ValueError(f'No model weights in {checkpoint_path}')
    model.load_state_dict(state)
    return model.eval()


def load_tokenizer(checkpoint_path: Path, device: torch.device):
    """Load either supported visual-token format without mixing code meanings."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('model_type') == 'pixel_patch_v1':
        from .pixel_patches import PixelPatchTokenizer
        model = PixelPatchTokenizer(checkpoint['codebook'])
        if (checkpoint['config']['num_embeddings'] != model.cfg.num_embeddings
                or checkpoint['config'].get('patch_size') != 2
                or checkpoint['config'].get('in_channels') != 3):
            raise ValueError('Patch vocabulary does not match checkpoint configuration')
        return model.to(device).eval()
    return _vqvae_from_checkpoint(checkpoint, device, checkpoint_path)
