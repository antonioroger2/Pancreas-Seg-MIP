"""
attention.py - 3D additive attention gates.

AttentionGate3D ("simple", the Review-1 gate; used for the R0-R4 ablation):
    theta = LN(Conv1x1(x) -> F_int), phi = LN(Conv1x1(g) -> F_int), f = ReLU(theta + phi),
    alpha = sigmoid(Conv1x1(f) -> 1), out = x * alpha                    (F_int = C/2)
PaperAttentionGate3D ("paper", optional flag --ag paper; paper Sec. 3.2.3, deviation D2):
    theta = LN(Conv1x1(x) -> C), phi = LN(Conv1x1(g) -> C), f = ReLU(theta + phi),
    r = ReLU(LN(Conv1x1(f) -> C/2)), alpha = sigmoid(Conv1x1(r) -> 1), out = LN(Conv1x1(x * alpha) -> C)
LN = parameter-free LayerNorm over (C, H, W, D): its statistics depend on the spatial size, so the ROI must be
identical in training and inference (R1-R4 use 128x128x96 for both).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ln(t):
    return F.layer_norm(t, t.shape[1:])


def _match(g, x):
    return g if g.shape[2:] == x.shape[2:] else F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=True)


class AttentionGate3D(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.conv_x = nn.Conv3d(F_l, F_int, 1, bias=False)
        self.conv_g = nn.Conv3d(F_g, F_int, 1, bias=False)
        self.conv_psi = nn.Conv3d(F_int, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = F.relu(_ln(self.conv_x(x)) + _ln(self.conv_g(_match(g, x))))
        return x * torch.sigmoid(self.conv_psi(f))


class PaperAttentionGate3D(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int = None):
        super().__init__()
        C = F_l
        self.conv_x = nn.Conv3d(F_l, C, 1, bias=False)
        self.conv_g = nn.Conv3d(F_g, C, 1, bias=False)
        self.reduce = nn.Conv3d(C, C // 2, 1, bias=False)
        self.conv_psi = nn.Conv3d(C // 2, 1, 1, bias=True)
        self.out = nn.Conv3d(C, C, 1, bias=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = F.relu(_ln(self.conv_x(x)) + _ln(self.conv_g(_match(g, x))))
        alpha = torch.sigmoid(self.conv_psi(F.relu(_ln(self.reduce(f)))))
        return _ln(self.out(x * alpha))
