import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib


def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_dice_score(pred_mask, target_mask, smooth=1e-5):
    """
    Computes binary Dice Similarity Coefficient (DSC).
    pred_mask and target_mask should be binary (0 or 1) PyTorch Tensors or Numpy arrays.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()

    pred_mask = (pred_mask > 0.5).astype(np.float32)
    target_mask = (target_mask > 0.5).astype(np.float32)

    intersection = np.sum(pred_mask * target_mask)
    total = np.sum(pred_mask) + np.sum(target_mask)

    if total == 0:
        return 1.0  # Perfect score if both ground truth and prediction are empty
    
    return (2.0 * intersection + smooth) / (total + smooth)


def plot_prediction_overlay(ct_slice, ground_truth, prediction, save_path=None, title="Pancreas Segmentation"):
    """
    Visualizes raw CT slice alongside Ground Truth mask (Green) and Predicted mask (Red/Cyan).
    """
    if isinstance(ct_slice, torch.Tensor):
        ct_slice = ct_slice.squeeze().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.squeeze().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CT Slice
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT Image")
    axes[0].axis("off")

    # Ground Truth Overlay
    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(ground_truth, cmap="Greens", alpha=0.5)
    axes[1].set_title("Ground Truth (Green)")
    axes[1].axis("off")

    # Prediction Overlay
    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(prediction > 0.5, cmap="Reds", alpha=0.5)
    axes[2].set_title(f"Prediction (Red) - {title}")
    axes[2].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def save_nifti_mask(mask_array, reference_nifti_path, output_save_path):
    """
    Saves predicted 3D mask array back to NIfTI format using reference metadata.
    """
    ref_img = nib.load(reference_nifti_path)
    mask_nifti = nib.Nifti1Image(mask_array.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(mask_nifti, output_save_path)
    print(f"Saved NIfTI mask to: {output_save_path}")


def plot_metrics(train_losses, val_losses, val_dices, save_dir="plots"):
    """
    Generates and saves publication-ready plots for Training/Validation Loss and Dice Coefficient.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # 1. Loss Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="royalblue", linewidth=2.5)
    plt.plot(epochs, val_losses, label="Val Loss", color="crimson", linewidth=2.5, linestyle="--")
    plt.title("Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice-CE Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path, bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Validation Dice Score Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_dices, label="Val Dice Score", color="forestgreen", linewidth=2.5, marker="o")
    plt.title("Validation Dice Similarity Coefficient (DSC)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    dice_path = os.path.join(save_dir, "dice_curve.png")
    plt.savefig(dice_path, bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Combined Summary Plot for Research Paper
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="crimson", linewidth=2, linestyle="--")
    ax1.set_title("Loss Curves", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, val_dices, label="Val Dice Score", color="forestgreen", linewidth=2, marker="o")
    ax2.set_title("Validation Dice Coefficient", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    summary_path = os.path.join(save_dir, "metrics_summary.png")
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved publication plots to: {save_dir}")

