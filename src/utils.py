"""
utils.py - Utility Functions for 3D Pancreas Segmentation.

Contains:
  - Random seed fixing
  - Volume-level Dice calculation
  - Visualization: prediction overlays, metric plots, 3D volume slices
  - NIfTI mask saving
  - Memory benchmarking
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib


def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_dice_score(pred_mask, target_mask, smooth=1e-5):
    """Compute binary Dice Similarity Coefficient (DSC)."""
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()

    pred_mask = (pred_mask > 0.5).astype(np.float32)
    target_mask = (target_mask > 0.5).astype(np.float32)

    intersection = np.sum(pred_mask * target_mask)
    total = np.sum(pred_mask) + np.sum(target_mask)

    if total == 0:
        return 1.0
    return (2.0 * intersection + smooth) / (total + smooth)


def plot_prediction_overlay(ct_slice, ground_truth, prediction,
                            save_path=None, title="Pancreas Segmentation"):
    """Visualize a single CT slice with GT and prediction overlays."""
    if isinstance(ct_slice, torch.Tensor):
        ct_slice = ct_slice.squeeze().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.squeeze().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT Image")
    axes[0].axis("off")

    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(ground_truth, cmap="Greens", alpha=0.5)
    axes[1].set_title("Ground Truth (Green)")
    axes[1].axis("off")

    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(prediction > 0.5, cmap="Reds", alpha=0.5)
    axes[2].set_title(f"Prediction (Red) - {title}")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_3d_volume_slices(image_vol, mask_vol, pred_vol=None,
                          save_path=None, n_slices=5, title="Volume Slices"):
    """Visualize multiple axial slices from a 3D volume."""
    D = image_vol.shape[2]
    slice_indices = np.linspace(0, D - 1, n_slices, dtype=int)

    n_cols = n_slices
    n_rows = 3 if pred_vol is not None else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, s_idx in enumerate(slice_indices):
        axes[0, col].imshow(image_vol[:, :, s_idx], cmap='gray')
        axes[0, col].set_title(f"Slice {s_idx}", fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(image_vol[:, :, s_idx], cmap='gray')
        axes[1, col].imshow(mask_vol[:, :, s_idx], cmap='Greens', alpha=0.5)
        axes[1, col].axis('off')

        if pred_vol is not None:
            axes[2, col].imshow(image_vol[:, :, s_idx], cmap='gray')
            axes[2, col].imshow(pred_vol[:, :, s_idx] > 0.5, cmap='Reds', alpha=0.5)
            axes[2, col].axis('off')

    axes[0, 0].set_ylabel("CT", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("GT", fontsize=12, fontweight='bold')
    if pred_vol is not None:
        axes[2, 0].set_ylabel("Pred", fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def save_nifti_mask(mask_array, reference_nifti_path, output_save_path):
    """Save predicted 3D mask array back to NIfTI format."""
    ref_img = nib.load(reference_nifti_path)
    mask_nifti = nib.Nifti1Image(mask_array.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(mask_nifti, output_save_path)
    print(f"Saved NIfTI mask to: {output_save_path}")


def plot_metrics(train_losses, val_losses, val_dices, save_dir="plots"):
    """Generate and save training metric plots (300 DPI)."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="royalblue", linewidth=2.5)
    plt.plot(epochs, val_losses, label="Val Loss", color="crimson", linewidth=2.5, linestyle="--")
    plt.title("Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice + Focal Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Dice Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_dices, label="Val Dice Score", color="forestgreen", linewidth=2.5, marker="o", markersize=3)
    plt.title("Validation Dice Similarity Coefficient (DSC)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(save_dir, "dice_curve.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Combined
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="crimson", linewidth=2, linestyle="--")
    ax1.set_title("Loss Curves", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6); ax1.legend()

    ax2.plot(epochs, val_dices, label="Val Dice", color="forestgreen", linewidth=2, marker="o", markersize=3)
    ax2.set_title("Validation Dice Coefficient", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice Score")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_summary.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved publication plots to: {save_dir}")


def memory_benchmark(device=None):
    """Benchmark GPU memory for one training step with paper-specified input size."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("GPU MEMORY BENCHMARK")
    print("=" * 60)

    if device.type != 'cuda':
        print("  No GPU available -- skipping benchmark")
        return None

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Total VRAM: {total_mem:.1f} GB")

    from src.models.model import build_model, count_parameters
    from src.losses import DiceFocalLoss
    from src.config import CROP_SIZE, USE_AMP

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = build_model().to(device)
    params = count_parameters(model)
    print(f"  Model parameters: {params['total']:,} ({params['total_MB']:.1f} MB)")

    criterion = DiceFocalLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    H, W, D = CROP_SIZE
    x = torch.randn(1, 1, H, W, D, device=device)
    y = torch.randint(0, 2, (1, 1, H, W, D), device=device).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    try:
        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
            output = model(x)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9

        print(f"\n  --- Results (224x224x128, batch=1, AMP={USE_AMP}) ---")
        print(f"  Allocated memory: {allocated:.2f} GB")
        print(f"  Peak memory:      {peak:.2f} GB")
        print(f"  Available margin: {total_mem - peak:.2f} GB")

        if peak < total_mem * 0.95:
            print(f"  [OK] FITS in GPU memory")
        else:
            print(f"  [WARN] TIGHT FIT -- consider gradient checkpointing")

        result = {
            'gpu_name': torch.cuda.get_device_name(0),
            'total_vram_gb': total_mem,
            'model_params': params['total'],
            'allocated_gb': allocated,
            'peak_gb': peak,
            'fits': peak < total_mem * 0.95,
            'amp': USE_AMP,
            'input_shape': (1, 1, H, W, D),
        }

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n  [FAIL] OUT OF MEMORY")
            result = {'fits': False, 'error': str(e)}
        else:
            raise

    torch.cuda.empty_cache()
    print(f"{'='*60}")
    return result


if __name__ == "__main__":
    memory_benchmark()
