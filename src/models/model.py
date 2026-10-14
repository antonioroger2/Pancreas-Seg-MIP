"""
model.py - 3D Attention U-Net (paper Fig. 1: channels 16-32-64-128-256) with ablation flags.

Flags (defaults = R0, the paper-faithful baseline; reproduces Review-1's 5,668,269 parameters exactly):
  norm      "batch" (BatchNorm3d, R0/R1) | "group" (GroupNorm(8, C), R2+; every C is divisible by 8)
  deepsup   3 auxiliary 1x1x1 heads on decoder outputs at 1/8, 1/4, 1/2 resolution, trilinear-upsampled to
            the input size. Train mode returns [main, aux_1/2, aux_1/4, aux_1/8]; eval mode returns main only.
  attention False -> skip connections pass through unchanged (plain 3D U-Net, R4)
  ag        "simple" (Review-1 gate, used in the ablation) | "paper" (Sec. 3.2.3 gate, optional, D2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ENCODER_CHANNELS, IN_CHANNELS, OUT_CHANNELS
from src.models.attention import AttentionGate3D, PaperAttentionGate3D


def norm_layer(c: int, norm: str) -> nn.Module:
    return nn.GroupNorm(8, c) if norm == "group" else nn.BatchNorm3d(c)


class DoubleConv3D(nn.Module):
    """(Conv3d 3x3x3, bias=False -> Norm -> ReLU) x 2."""

    def __init__(self, i: int, o: int, norm: str = "batch"):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(i, o, 3, padding=1, bias=False), norm_layer(o, norm), nn.ReLU(inplace=True),
            nn.Conv3d(o, o, 3, padding=1, bias=False), norm_layer(o, norm), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=None,
                 norm="batch", deepsup=False, attention=True, ag="simple"):
        super().__init__()
        ch = list(channels or ENCODER_CHANNELS)
        self.deepsup, self.attention = deepsup, attention
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.enc.append(DoubleConv3D(prev, c, norm)); prev = c
        self.pool = nn.MaxPool3d(2)
        rev = ch[::-1]                                              # 256, 128, 64, 32, 16
        Gate = PaperAttentionGate3D if ag == "paper" else AttentionGate3D
        self.up = nn.ModuleList(nn.ConvTranspose3d(rev[i], rev[i + 1], 2, stride=2) for i in range(len(rev) - 1))
        self.gates = nn.ModuleList(Gate(rev[i + 1], rev[i + 1], rev[i + 1] // 2) for i in range(len(rev) - 1)) \
            if attention else None
        self.dec = nn.ModuleList(DoubleConv3D(2 * rev[i + 1], rev[i + 1], norm) for i in range(len(rev) - 1))
        self.final_conv = nn.Conv3d(ch[0], out_channels, 1)
        # aux heads on decoder levels 0..2 (1/8, 1/4, 1/2 resolution)
        self.aux = nn.ModuleList(nn.Conv3d(rev[i + 1], out_channels, 1) for i in range(3)) if deepsup else None

    def forward(self, x):
        size, skips = x.shape[2:], []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x); x = self.pool(x)
        dec_out = []
        for i, (up, conv) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[-1 - i]
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:], mode="trilinear", align_corners=True)
            if self.attention:
                s = self.gates[i](x=s, g=x)
            x = conv(torch.cat([s, x], dim=1))
            dec_out.append(x)
        main = self.final_conv(x)
        if not (self.deepsup and self.training):
            return main
        aux = [F.interpolate(h(d), size=size, mode="trilinear", align_corners=False)
               for h, d in zip(self.aux, dec_out[:3])]                 # [1/8, 1/4, 1/2]
        return [main, aux[2], aux[1], aux[0]]                           # weights 1, 1/2, 1/4, 1/8


def build_model(norm="batch", deepsup=False, attention=True, ag="simple", encoder_channels=None, **_) -> AttentionUNet3D:
    """Extra keys (e.g. a full run config) are ignored, so build_model(**config) works."""
    return AttentionUNet3D(channels=encoder_channels, norm=norm, deepsup=deepsup, attention=attention, ag=ag)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_MB": total * 4 / 1024 ** 2}
