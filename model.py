import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from monai.networks.nets import UNet as MonaiUNet
except ImportError:
    MonaiUNet = None


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet2D(nn.Module):
    """
    Standard 2D U-Net for Medical Image Segmentation (Pancreas CT Slices).
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        current_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final Output Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


class DiceCELoss(nn.Module):
    """
    Combined Binary Cross Entropy (BCE) + Soft Dice Loss for highly imbalanced binary segmentation.
    """
    def __init__(self, smooth=1e-5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def build_model(model_type="2d", in_channels=1, out_channels=1, light=True):
    """
    Factory function to instantiate U-Net architecture.
    Setting light=True uses features=[32, 64, 128, 256] for fast 30-min training on Colab T4 GPUs.
    """
    features = [32, 64, 128, 256] if light else [64, 128, 256, 512]
    
    if model_type == "2d":
        return UNet2D(in_channels=in_channels, out_channels=out_channels, features=features)
    elif model_type == "3d_monai" and MonaiUNet is not None:
        channels = (16, 32, 64, 128) if light else (16, 32, 64, 128, 256)
        strides = (2, 2, 2) if light else (2, 2, 2, 2)
        return MonaiUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
        )
    else:
        return UNet2D(in_channels=in_channels, out_channels=out_channels, features=features)

