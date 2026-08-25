"""
model.py - 3D Attention U-Net for Pancreas Segmentation.

Implements the paper's architecture (Section 2.1, Figure 1):
  - 5-level 3D U-Net encoder with channels [16, 32, 64, 128, 256]
  - Attention Gates at every skip connection
  - 3x3x3 convolutions, BatchNorm3d, ReLU
  - MaxPool3d(2) downsampling, ConvTranspose3d(2) upsampling
  - 1x1x1 final convolution for binary segmentation

PAPER AMBIGUITY #1 -- Channel Count:
  Text: "starting from 2 to 16" -- inconsistent with any viable architecture.
  Figure 1: clearly shows 16 -> 32 -> 64 -> 128 -> 256.
  RESOLUTION: Follow Figure 1.

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionGate3D
from config import ENCODER_CHANNELS, IN_CHANNELS, OUT_CHANNELS


class DoubleConv3D(nn.Module):
    """
    Double 3D Convolution Block:
        Conv3d(3x3x3) -> BatchNorm3d -> ReLU -> Conv3d(3x3x3) -> BatchNorm3d -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Encoder3D(nn.Module):
    """
    3D U-Net Encoder -- 5 resolution levels with MaxPool3d downsampling.
    """

    def __init__(self, in_channels: int = IN_CHANNELS, channels: list = None):
        super().__init__()
        if channels is None:
            channels = ENCODER_CHANNELS

        self.levels = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        prev_ch = in_channels
        for ch in channels:
            self.levels.append(DoubleConv3D(prev_ch, ch))
            prev_ch = ch

    def forward(self, x: torch.Tensor):
        features = []
        for i, level in enumerate(self.levels):
            x = level(x)
            if i < len(self.levels) - 1:
                features.append(x)
                x = self.pool(x)
        return features, x


class Decoder3D(nn.Module):
    """
    3D U-Net Decoder with Attention Gates at skip connections.
    """

    def __init__(self, channels: list = None):
        super().__init__()
        if channels is None:
            channels = ENCODER_CHANNELS

        self.up_convs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.double_convs = nn.ModuleList()

        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]

            self.up_convs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate3D(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2 if out_ch > 1 else 1)
            )
            self.double_convs.append(
                DoubleConv3D(out_ch * 2, out_ch)
            )

    def forward(self, features: list, bottleneck: torch.Tensor) -> torch.Tensor:
        x = bottleneck
        reversed_features = list(reversed(features))

        for i, (up, ag, conv) in enumerate(zip(self.up_convs,
                                                self.attention_gates,
                                                self.double_convs)):
            x = up(x)
            skip = reversed_features[i]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)

            skip_attended = ag(x=skip, g=x)
            x = torch.cat([skip_attended, x], dim=1)
            x = conv(x)

        return x


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net -- complete model.
    Channels follow Figure 1: 16 -> 32 -> 64 -> 128 -> 256
    """

    def __init__(self, in_channels: int = IN_CHANNELS,
                 out_channels: int = OUT_CHANNELS,
                 encoder_channels: list = None):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = ENCODER_CHANNELS

        self.encoder = Encoder3D(in_channels, encoder_channels)
        self.decoder = Decoder3D(encoder_channels)
        self.final_conv = nn.Conv3d(encoder_channels[0], out_channels,
                                     kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, bottleneck = self.encoder(x)
        decoded = self.decoder(features, bottleneck)
        output = self.final_conv(decoded)
        return output


def build_model(in_channels: int = IN_CHANNELS,
                out_channels: int = OUT_CHANNELS,
                encoder_channels: list = None) -> AttentionUNet3D:
    """Factory function to build the 3D Attention U-Net."""
    return AttentionUNet3D(in_channels, out_channels, encoder_channels)


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_MB': total * 4 / (1024 ** 2),
    }
