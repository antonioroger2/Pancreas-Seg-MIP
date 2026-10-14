"""
losses.py - Dice + Focal loss (paper Eq. 2-3) and the deep-supervision wrapper.

Focal (paper Eq. 2): -[g (1-p)^gamma log p + (1-g) p^gamma log(1-p)], mean over voxels, gamma = 2,
NO alpha term (decision D1). alpha (e.g. 0.25 = Review-1 setting) only via --focal_alpha.
Dice: 1 - (2 sum(pg) + eps) / (sum p + sum g + eps), per sample, eps = 1e-5 (standard form; Eq. 3 lacks the 2).
Losses are computed in float32 even under AMP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DICE_SMOOTH, FOCAL_ALPHA, FOCAL_GAMMA


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = DICE_SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits.float()).flatten(1)
        t = target.float().flatten(1)
        dice = (2 * (p * t).sum(1) + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        logits, t = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")    # = -log p_t (stable)
        p = torch.sigmoid(logits)
        p_t = p * t + (1 - p) * (1 - t)
        w = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            w = w * (self.alpha * t + (1 - self.alpha) * (1 - t))
        return (w * bce).mean()


class DiceFocalLoss(nn.Module):
    """Paper loss: L = Dice + Focal."""

    def __init__(self, dice_smooth=DICE_SMOOTH, focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA):
        super().__init__()
        self.dice, self.focal = DiceLoss(dice_smooth), FocalLoss(focal_gamma, focal_alpha)

    def forward(self, logits, target):
        return self.dice(logits, target) + self.focal(logits, target)


class DeepSupervisionLoss(nn.Module):
    """sum_i w_i * crit(out_i, y), w = [1, 1/2, 1/4, 1/8] / 1.875; a single tensor falls back to crit."""

    def __init__(self, crit: nn.Module, weights=(1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32)
        self.crit = crit
        self.register_buffer("w", w / w.sum())

    def forward(self, outs, target):
        if torch.is_tensor(outs):
            return self.crit(outs, target)
        return sum(wi * self.crit(o, target) for wi, o in zip(self.w, outs))
