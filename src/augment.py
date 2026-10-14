"""
augment.py - ONE GPU augmentation used identically by every run (decision D7, paper Sec. 2.2):
rotation +-10 deg about each axis, flip each axis with p = 0.5, translation <= 10 % of the size.
Image: trilinear (border padding); label: nearest (zero padding), re-binarised.
Rotation is applied in normalised coordinates, so on non-cubic patches +-10 deg adds a negligible shear.
Flipping along S-I / A-P is anatomically implausible but paper-faithful.
"""
import math

import torch
import torch.nn.functional as F

from src.config import AUG_FLIP_PROB, AUG_ROTATION_RANGE, AUG_SHIFT_RANGE


def gpu_augment(x, y, rot_deg=AUG_ROTATION_RANGE, shift=AUG_SHIFT_RANGE, flip_p=AUG_FLIP_PROB):
    """x, y: (B, 1, H, W, D) float tensors on the same device. One random affine per sample."""
    B, dev = x.shape[0], x.device
    a = (torch.rand(B, 3, device=dev) * 2 - 1) * rot_deg * math.pi / 180
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(c[:, 0]), torch.zeros_like(c[:, 0])
    Rx = torch.stack([o, z, z, z, c[:, 0], -s[:, 0], z, s[:, 0], c[:, 0]], 1).view(B, 3, 3)
    Ry = torch.stack([c[:, 1], z, s[:, 1], z, o, z, -s[:, 1], z, c[:, 1]], 1).view(B, 3, 3)
    Rz = torch.stack([c[:, 2], -s[:, 2], z, s[:, 2], c[:, 2], z, z, z, o], 1).view(B, 3, 3)
    flip = torch.where(torch.rand(B, 3, device=dev) < flip_p, -1.0, 1.0)
    A = (Rz @ Ry @ Rx) * flip[:, None, :]                          # flip = negate columns
    t = (torch.rand(B, 3, 1, device=dev) * 2 - 1) * shift * 2      # normalised coordinates span 2
    grid = F.affine_grid(torch.cat([A, t], 2), list(x.shape), align_corners=False)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    yo = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return xo, (yo > 0.5).float()
