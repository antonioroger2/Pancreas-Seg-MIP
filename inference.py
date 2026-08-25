"""
inference.py - Inference on New CT Volumes.

Loads a trained 3D Attention U-Net checkpoint and runs inference
on preprocessed CT volumes. Outputs predicted segmentation masks.

Usage:
    python inference.py \
        --input_dir ./data/preprocessed/images \
        --checkpoint ./checkpoints/fold_0/best_model.pth \
        --output_dir ./predictions

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from config import (
    ENCODER_CHANNELS, IN_CHANNELS, OUT_CHANNELS,
    EVAL_THRESHOLD, TARGET_SPACING, USE_AMP,
)
from model import build_model
from preprocessing import preprocess_volume


@torch.no_grad()
def run_inference(model, image_path: str, device: torch.device,
                  preprocessed: bool = True) -> np.ndarray:
    """
    Run inference on a single volume.

    Args:
        model: trained model in eval mode
        image_path: path to NIfTI image
        device: torch device
        preprocessed: if True, volume is already preprocessed

    Returns:
        Binary prediction mask (H, W, D)
    """
    model.eval()

    if preprocessed:
        img_nii = nib.load(image_path)
        image = img_nii.get_fdata().astype(np.float32)
    else:
        result = preprocess_volume(image_path)
        image = result['image']

    # To tensor: (1, 1, H, W, D)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
        output = model(image_tensor)

    pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred_prob > EVAL_THRESHOLD).astype(np.uint8)

    return pred_mask


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained 3D Attention U-Net"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing NIfTI images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="Output directory for predicted masks")
    parser.add_argument("--threshold", type=float, default=EVAL_THRESHOLD,
                        help=f"Binarization threshold (default: {EVAL_THRESHOLD})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(IN_CHANNELS, OUT_CHANNELS, ENCODER_CHANNELS)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_val_dice' in checkpoint:
        print(f"  Checkpoint best Dice: {checkpoint['best_val_dice']:.4f}")

    # Find input images
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.nii*")))
    if not image_paths:
        print(f"No NIfTI files found in {args.input_dir}")
        return

    print(f"\nFound {len(image_paths)} volumes to process")

    # Run inference
    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Inference"):
        basename = os.path.basename(img_path)
        output_name = basename.replace('.nii', '_pred.nii')

        pred_mask = run_inference(model, img_path, device, preprocessed=True)

        # Save as NIfTI
        affine = np.eye(4)
        affine[0, 0], affine[1, 1], affine[2, 2] = TARGET_SPACING
        pred_nifti = nib.Nifti1Image(pred_mask, affine)
        output_path = os.path.join(args.output_dir, output_name)
        nib.save(pred_nifti, output_path)

    print(f"\nInference complete. Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
