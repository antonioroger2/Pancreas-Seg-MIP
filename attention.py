"""
attention.py - 3D Attention Gate Module for Attention U-Net.

Implements the paper's attention mechanism (Section 2.1, Figure 1):
    theta(x) = LayerNorm(Conv3d_1x1(x))
    phi(g) = LayerNorm(Conv3d_1x1(g))
    f = ReLU(theta(x) + phi(g))
    psi = Conv3d_1x1(f) -> 1 channel
    alpha = Sigmoid(psi) -> attention map
    x' = x * alpha -> modulated features

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate3D(nn.Module):
    """
    3D Additive Attention Gate for skip-connection modulation.

    Takes encoder features (skip connection) and decoder gating signal,
    computes an attention map alpha in [0, 1], and modulates the skip features.

    Args:
        F_g: number of channels in the gating signal (from decoder)
        F_l: number of channels in the skip connection (from encoder)
        F_int: number of intermediate channels for attention computation
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # theta(x): transform skip connection features
        self.conv_x = nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False)

        # phi(g): transform gating signal
        self.conv_g = nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False)

        # psi: reduce to single-channel attention map
        self.conv_psi = nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.F_int = F_int
        self.F_l = F_l

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Attention Gate.

        Args:
            x: encoder skip features, shape (B, F_l, H, W, D)
            g: decoder gating signal, shape (B, F_g, H', W', D')

        Returns:
            Attention-modulated features, shape (B, F_l, H, W, D)
        """
        # Ensure gating signal matches spatial dimensions of skip connection
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=True)

        # theta(x): transform skip features + layer norm
        theta_x = self.conv_x(x)
        theta_x = F.layer_norm(theta_x, theta_x.shape[1:])

        # phi(g): transform gating signal + layer norm
        phi_g = self.conv_g(g)
        phi_g = F.layer_norm(phi_g, phi_g.shape[1:])

        # Additive attention: f = ReLU(theta(x) + phi(g))
        f = self.relu(theta_x + phi_g)

        # psi: reduce to attention coefficients
        psi = self.conv_psi(f)
        alpha = self.sigmoid(psi)  # (B, 1, H, W, D)

        # Modulate skip features
        out = x * alpha

        return out


def test_attention_gate():
    """Test attention gate with random tensors."""
    print("=" * 60)
    print("ATTENTION GATE UNIT TEST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_configs = [
        (256, 128, 128, (14, 14, 8), (7, 7, 4), "Level 4: bottleneck->enc4"),
        (128, 64, 64, (28, 28, 16), (14, 14, 8), "Level 3: dec4->enc3"),
        (64, 32, 32, (56, 56, 32), (28, 28, 16), "Level 2: dec3->enc2"),
        (32, 16, 16, (112, 112, 64), (56, 56, 32), "Level 1: dec2->enc1"),
    ]

    all_passed = True

    for F_g, F_l, F_int, spatial_x, spatial_g, desc in test_configs:
        print(f"\n--- {desc} ---")

        ag = AttentionGate3D(F_g, F_l, F_int).to(device)
        x = torch.randn(1, F_l, *spatial_x, device=device, requires_grad=True)
        g = torch.randn(1, F_g, *spatial_g, device=device, requires_grad=True)

        try:
            out = ag(x, g)
            assert out.shape == x.shape, f"Shape mismatch: out {out.shape} vs x {x.shape}"

            loss = out.sum()
            loss.backward()
            assert x.grad is not None, "No gradient for skip connection"
            assert g.grad is not None, "No gradient for gating signal"

            print(f"  [OK] Output shape: {out.shape}")
            print(f"  [OK] Gradients flow correctly")
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            all_passed = False

    print(f"\n{'='*60}")
    print("ALL ATTENTION GATE TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print(f"{'='*60}")
    return all_passed


if __name__ == "__main__":
    test_attention_gate()
