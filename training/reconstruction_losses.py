"""Local reconstruction objectives; no post-processing or artificial sharpening."""
import torch
import torch.nn.functional as F


def signed_gradient_loss(reconstruction: torch.Tensor, target: torch.Tensor):
    """Match signed horizontal/vertical differences, including flat regions.

    Signed differences penalize reversed edges as well as blurred edges.
    Compute in float32 for stable mixed-precision training gradients.
    Pixel reconstruction loss remains necessary to constrain absolute colors.
    """
    if reconstruction.shape != target.shape or target.ndim != 4:
        raise ValueError('Expected matching NCHW images')
    if min(target.shape[-2:]) < 2:
        raise ValueError('Spatial dimensions must be at least 2')
    p,t=reconstruction.float(),target.float()
    horizontal=F.l1_loss(p[:,:,:,1:]-p[:,:,:,:-1],t[:,:,:,1:]-t[:,:,:,:-1])
    vertical=F.l1_loss(p[:,:,1:,:]-p[:,:,:-1,:],t[:,:,1:,:]-t[:,:,:-1,:])
    return (horizontal+vertical)*.5
