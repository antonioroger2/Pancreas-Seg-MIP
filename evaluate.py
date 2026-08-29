"""
evaluate.py - Volume-Level Evaluation for 3D Pancreas Segmentation.

Computes the paper's evaluation metrics (Section 2.4):
  - Volumetric DSC (Dice Similarity Coefficient)
  - ASSD (Average Symmetric Surface Distance) in mm
  - HD95 (95th Percentile Hausdorff Distance) in mm

Supports evaluation on:
  - Individual folds (validation sets)
  - Independent test set
  - All folds aggregated

Saves:
  - Per-patient CSV results
  - Aggregate summary
  - NIfTI prediction masks

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import os
import csv
import json
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

from config import (
    ENCODER_CHANNELS, IN_CHANNELS, OUT_CHANNELS,
    EVAL_THRESHOLD, TARGET_SPACING, CROP_SIZE,
    DEFAULT_DATA_DIR, DEFAULT_CHECKPOINT_DIR, DEFAULT_RESULTS_DIR,
    DEFAULT_SPLITS_DIR, USE_AMP,
)
from model import build_model
from dataset import discover_data_paths, PancreasVolumeDataset
from metrics import compute_all_metrics, aggregate_metrics


def keep_largest_components(binary_mask: np.ndarray, n_components: int = 2):
    """
    Keep only the n largest connected components of a binary mask.
    Standard pancreas-segmentation post-processing to remove scattered
    false-positive fragments. Falls back to the raw mask if empty.
    """
    from scipy import ndimage
    if binary_mask.sum() == 0:
        return binary_mask
    lbl, num = ndimage.label(binary_mask)
    if num <= n_components:
        return binary_mask
    sizes = ndimage.sum(binary_mask, lbl, range(1, num + 1))
    keep = np.argsort(sizes)[-n_components:] + 1
    return np.isin(lbl, keep).astype(binary_mask.dtype)


@torch.no_grad()
def evaluate_volumes(model, image_paths: list, mask_paths: list,
                     device: torch.device, save_preds_dir: str = None,
                     voxel_spacing: tuple = TARGET_SPACING,
                     postprocess: bool = False) -> list:
    """
    Evaluate model on a set of volumes. Returns per-patient metrics.

    Args:
        model: trained model (in eval mode)
        image_paths: list of image NIfTI paths
        mask_paths: list of corresponding mask NIfTI paths
        device: torch device
        save_preds_dir: if set, save predicted masks as NIfTI here
        voxel_spacing: physical voxel spacing for distance metrics
        postprocess: if True, keep only the 2 largest connected components

    Returns:
        list of dicts with keys: patient_id, dice, assd, hd95
    """
    model.eval()

    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)

    dataset = PancreasVolumeDataset(image_paths, mask_paths,
                                     augment=False, preprocessed=True)
    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        image_tensor, mask_tensor = dataset[idx]

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Forward pass
        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            output = model(image_tensor)

        # Get prediction
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        gt_mask = mask_tensor.squeeze().cpu().numpy()

        # Optional largest-connected-component post-processing
        if postprocess:
            pred_bin_raw = (pred_prob > EVAL_THRESHOLD).astype(np.uint8)
            pred_bin = keep_largest_components(pred_bin_raw, n_components=2)
            pred_prob = pred_bin.astype(np.float32)

        # Compute metrics
        patient_id = dataset.get_patient_id(idx)
        m = compute_all_metrics(pred_prob, gt_mask, voxel_spacing, EVAL_THRESHOLD)
        m['patient_id'] = patient_id
        results.append(m)

        print(f"  Patient {patient_id}: DSC={m['dice']:.4f}, "
              f"ASSD={m['assd']:.2f} mm, HD95={m['hd95']:.2f} mm")

        # Save prediction mask as NIfTI
        if save_preds_dir:
            pred_bin = (pred_prob > EVAL_THRESHOLD).astype(np.uint8)
            affine = np.eye(4)
            affine[0, 0], affine[1, 1], affine[2, 2] = voxel_spacing
            pred_nifti = nib.Nifti1Image(pred_bin, affine)
            pred_path = os.path.join(save_preds_dir,
                                      f"pred_PANCREAS_{patient_id}.nii.gz")
            nib.save(pred_nifti, pred_path)

    return results


def save_results(results: list, output_dir: str, prefix: str = ""):
    """Save per-patient results and aggregate summary to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-patient CSV
    per_patient_path = os.path.join(output_dir, f"{prefix}per_patient_results.csv")
    with open(per_patient_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'dice', 'assd_mm', 'hd95_mm'])
        for r in results:
            writer.writerow([
                r['patient_id'],
                f"{r['dice']:.6f}",
                f"{r['assd']:.4f}" if r['assd'] != float('inf') else "inf",
                f"{r['hd95']:.4f}" if r['hd95'] != float('inf') else "inf",
            ])
    print(f"  Per-patient results: {per_patient_path}")

    # Aggregate summary
    agg = aggregate_metrics(results)
    summary_path = os.path.join(output_dir, f"{prefix}summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'min', 'max'])
        writer.writerow(['DSC', f"{agg['dice_mean']:.4f}", f"{agg['dice_std']:.4f}",
                         f"{agg['dice_min']:.4f}", f"{agg['dice_max']:.4f}"])
        if agg['assd_mean'] != float('inf'):
            writer.writerow(['ASSD_mm', f"{agg['assd_mean']:.4f}", f"{agg['assd_std']:.4f}",
                             "", ""])
        if agg['hd95_mean'] != float('inf'):
            writer.writerow(['HD95_mm', f"{agg['hd95_mean']:.4f}", f"{agg['hd95_std']:.4f}",
                             "", ""])
    print(f"  Summary: {summary_path}")

    # Print summary
    print(f"\n  ┌{'─'*40}┐")
    print(f"  │ DSC:  {agg['dice_mean']:.4f} ± {agg['dice_std']:.4f}       │")
    print(f"  │ ASSD: {agg['assd_mean']:.4f} ± {agg['assd_std']:.4f} mm    │")
    print(f"  │ HD95: {agg['hd95_mean']:.4f} ± {agg['hd95_std']:.4f} mm    │")
    print(f"  └{'─'*40}┘")

    return agg


def evaluate_fold(fold: int, data_dir: str, checkpoint_dir: str,
                  splits_dir: str, results_dir: str, postprocess: bool = False):
    """Evaluate a trained fold on its validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    fold_ckpt = os.path.join(checkpoint_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(fold_ckpt):
        print(f"[WARNING] No checkpoint for fold {fold}: {fold_ckpt}")
        return None

    model = build_model(IN_CHANNELS, OUT_CHANNELS, ENCODER_CHANNELS)
    checkpoint = torch.load(fold_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load split
    fold_file = os.path.join(splits_dir, f"fold_{fold}.json")
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)

    # Get paths
    image_paths, mask_paths = discover_data_paths(data_dir)
    val_images = [image_paths[i] for i in fold_data['val_indices']]
    val_masks = [mask_paths[i] for i in fold_data['val_indices']]

    print(f"\n--- Evaluating Fold {fold} ({len(val_images)} patients) ---")

    # Evaluate
    fold_results_dir = os.path.join(results_dir, f"fold_{fold}")
    preds_dir = os.path.join(fold_results_dir, "predictions")

    results = evaluate_volumes(model, val_images, val_masks, device,
                               save_preds_dir=preds_dir, postprocess=postprocess)
    agg = save_results(results, fold_results_dir, prefix=f"fold_{fold}_")

    return results, agg


def evaluate_test_set(data_dir: str, checkpoint_dir: str,
                      splits_dir: str, results_dir: str,
                      use_best_fold: int = None, postprocess: bool = False):
    """Evaluate on the independent test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load splits
    splits = json.load(open(os.path.join(splits_dir, "patient_splits.json")))
    test_indices = splits['test_indices']

    if len(test_indices) == 0:
        print("[WARNING] No test set defined in splits")
        return None

    # Select which fold's model to use
    if use_best_fold is not None:
        fold_ckpt = os.path.join(checkpoint_dir, f"fold_{use_best_fold}", "best_model.pth")
    else:
        # Use fold 0 by default
        fold_ckpt = os.path.join(checkpoint_dir, "fold_0", "best_model.pth")

    if not os.path.exists(fold_ckpt):
        print(f"[WARNING] No checkpoint found: {fold_ckpt}")
        return None

    model = build_model(IN_CHANNELS, OUT_CHANNELS, ENCODER_CHANNELS)
    checkpoint = torch.load(fold_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get test paths
    image_paths, mask_paths = discover_data_paths(data_dir)
    test_images = [image_paths[i] for i in test_indices]
    test_masks = [mask_paths[i] for i in test_indices]

    print(f"\n--- Evaluating Independent Test Set ({len(test_images)} patients) ---")

    test_results_dir = os.path.join(results_dir, "test_set")
    preds_dir = os.path.join(test_results_dir, "predictions")

    results = evaluate_volumes(model, test_images, test_masks, device,
                               save_preds_dir=preds_dir, postprocess=postprocess)
    agg = save_results(results, test_results_dir, prefix="test_")

    return results, agg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D Pancreas Segmentation Model"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--splits_dir", type=str, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--fold", type=int, default=None,
                        help="Evaluate specific fold. Omit to evaluate all.")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate on independent test set")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all folds + test set")
    parser.add_argument("--postprocess", action="store_true",
                        help="Keep 2 largest connected components of predictions")
    args = parser.parse_args()

    if args.all or (args.fold is None and not args.test):
        # Evaluate all folds
        splits = json.load(open(os.path.join(args.splits_dir, "patient_splits.json")))
        for fold_data in splits['folds']:
            evaluate_fold(fold_data['fold'], args.data_dir, args.checkpoint_dir,
                         args.splits_dir, args.results_dir, postprocess=args.postprocess)
        # Test set
        evaluate_test_set(args.data_dir, args.checkpoint_dir,
                         args.splits_dir, args.results_dir, postprocess=args.postprocess)
    elif args.test:
        evaluate_test_set(args.data_dir, args.checkpoint_dir,
                         args.splits_dir, args.results_dir, postprocess=args.postprocess)
    elif args.fold is not None:
        evaluate_fold(args.fold, args.data_dir, args.checkpoint_dir,
                     args.splits_dir, args.results_dir, postprocess=args.postprocess)


if __name__ == "__main__":
    main()
