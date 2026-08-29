"""
losses.py - Loss Functions for 3D Pancreas Segmentation.

Implements the paper's combined Dice + Focal loss (Section 2.3):
    L_total = L_Dice + L_Focal

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DICE_SMOOTH, FOCAL_GAMMA, FOCAL_ALPHA


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Standard formulation with 2x in numerator.
    """

    def __init__(self, smooth: float = DICE_SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        pred_flat = pred.contiguous().view(pred.shape[0], -1)
        target_flat = target.contiguous().view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pred_prob = torch.sigmoid(pred_logits)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss -- the paper's primary loss function.
    L_total = L_Dice + L_Focal
    """

    def __init__(self, dice_smooth: float = DICE_SMOOTH,
                 focal_gamma: float = FOCAL_GAMMA,
                 focal_alpha: float = FOCAL_ALPHA,
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d_loss = self.dice_loss(pred_logits, target)
        f_loss = self.focal_loss(pred_logits, target)
        return self.dice_weight * d_loss + self.focal_weight * f_loss


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss (legacy from original codebase).
    NOT used in paper reproduction -- use DiceFocalLoss instead.
    """

    def __init__(self, smooth: float = DICE_SMOOTH, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred_logits, target)
        pred_prob = torch.sigmoid(pred_logits)
        pred_flat = pred_prob.contiguous().view(pred_prob.shape[0], -1)
        target_flat = target.contiguous().view(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
