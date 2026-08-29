"""
train.py - 3D Training Loop for Attention U-Net Pancreas Segmentation.

Implements:
  - Adam optimizer with LR = 6e-4 (paper Section 2.3)
  - ReduceLROnPlateau scheduler (paper: patience = 100)
  - Dice + Focal loss (paper Section 2.3)
  - AMP mixed precision for T4 GPU efficiency
  - Checkpoint/resume for Colab session recovery
  - Volume-level validation Dice
  - Per-epoch metrics logging

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import os
import argparse
import csv
import json
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR, DEFAULT_EPOCHS, BATCH_SIZE, USE_AMP, SEED,
    ENCODER_CHANNELS, IN_CHANNELS, OUT_CHANNELS, EVAL_THRESHOLD,
    FOCAL_GAMMA, FOCAL_ALPHA, DICE_SMOOTH,
    DEFAULT_DATA_DIR, DEFAULT_CHECKPOINT_DIR,
)
from src.models.model import build_model, count_parameters
from src.losses import DiceFocalLoss, DiceBCELoss, DiceLoss
from src.data.dataset import create_simple_dataloaders, create_fold_dataloaders
from src.utils import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch. Returns average training loss."""
    model.train()
    running_loss = 0.0
    n_samples = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate and compute volume-level Dice. Returns avg loss, avg Dice."""
    model.eval()
    running_loss = 0.0
    dice_scores = []
    n_samples = 0

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        # Volume-level Dice
        preds = torch.sigmoid(outputs)
        preds_bin = (preds > EVAL_THRESHOLD).float()

        for p, l in zip(preds_bin, labels):
            intersection = (p * l).sum()
            total = p.sum() + l.sum()
            if total == 0:
                dice_scores.append(1.0)
            else:
                dice_scores.append((2.0 * intersection / total).item())

    val_loss = running_loss / max(n_samples, 1)
    val_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return val_loss, val_dice


def save_checkpoint(state: dict, filepath: str, drive_path: str = None):
    """Save checkpoint locally and optionally to Google Drive."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    torch.save(state, filepath)

    if drive_path:
        os.makedirs(os.path.dirname(drive_path) or '.', exist_ok=True)
        torch.save(state, drive_path)


def load_checkpoint(filepath: str, model, optimizer, scheduler, scaler, device):
    """
    Load checkpoint and restore training state.
    Returns the epoch to resume from and best validation Dice.
    """
    if not os.path.exists(filepath):
        return 0, 0.0

    print(f"[Resume] Loading checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('best_val_dice', 0.0)

    print(f"[Resume] Resuming from epoch {start_epoch + 1}, best Dice: {best_dice:.4f}")
    return start_epoch, best_dice


def train(data_dir: str, checkpoint_dir: str, drive_checkpoint_dir: str = None,
          fold: int = None, train_indices: list = None, val_indices: list = None,
          epochs: int = DEFAULT_EPOCHS, batch_size: int = BATCH_SIZE,
          lr: float = LEARNING_RATE, resume: bool = True,
          num_workers: int = 2, preprocessed: bool = True,
          loss: str = 'dicefocal', focal_gamma: float = None,
          focal_alpha: float = None):
    """
    Main training function.

    Args:
        data_dir: directory with preprocessed images/ and labels/
        checkpoint_dir: local checkpoint directory
        drive_checkpoint_dir: Google Drive checkpoint directory (optional)
        fold: cross-validation fold number (for naming)
        train_indices: patient indices for training (if using CV)
        val_indices: patient indices for validation (if using CV)
        epochs: total number of epochs
        batch_size: batch size (default 1)
        lr: learning rate
        resume: whether to resume from checkpoint
        num_workers: DataLoader workers
        preprocessed: whether volumes are already preprocessed

    Returns:
        dict with training results (best_dice, train_losses, val_losses, etc.)
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[CPU] No GPU available — training will be slow")

    # Create checkpoint directory
    fold_str = f"fold_{fold}" if fold is not None else "single"
    ckpt_dir = os.path.join(checkpoint_dir, fold_str)
    os.makedirs(ckpt_dir, exist_ok=True)

    drive_ckpt_dir = None
    if drive_checkpoint_dir:
        drive_ckpt_dir = os.path.join(drive_checkpoint_dir, fold_str)
        os.makedirs(drive_ckpt_dir, exist_ok=True)

    # Create DataLoaders
    print("\n[Data] Loading dataset...")
    if train_indices is not None and val_indices is not None:
        train_loader, val_loader = create_fold_dataloaders(
            data_dir, train_indices, val_indices,
            batch_size=batch_size, num_workers=num_workers,
            preprocessed=preprocessed
        )
    else:
        train_loader, val_loader = create_simple_dataloaders(
            data_dir, batch_size=batch_size,
            num_workers=num_workers, preprocessed=preprocessed
        )

    # Build model
    print("\n[Model] Building 3D Attention U-Net...")
    model = build_model(IN_CHANNELS, OUT_CHANNELS, ENCODER_CHANNELS)
    model = model.to(device)

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # Loss, optimizer, scheduler
    if loss == 'dice':
        criterion = DiceLoss(smooth=DICE_SMOOTH)
    elif loss == 'dicebce':
        criterion = DiceBCELoss(smooth=DICE_SMOOTH, bce_weight=0.5)
    elif loss == 'dicefocal':
        gamma = FOCAL_GAMMA if focal_gamma is None else focal_gamma
        alpha = FOCAL_ALPHA if focal_alpha is None else focal_alpha
        criterion = DiceFocalLoss(
            dice_smooth=DICE_SMOOTH,
            focal_gamma=gamma,
            focal_alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown loss: {loss}")

    # Paper: Adam optimizer, LR = 6e-4 (see config.py for ambiguity note)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Paper: ReduceLROnPlateau with patience = 100
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
    )

    # AMP Scaler
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP and device.type == 'cuda')

    # Resume from checkpoint
    start_epoch = 0
    best_val_dice = 0.0
    resume_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")

    if resume:
        # Try drive checkpoint first, then local
        if drive_ckpt_dir and os.path.exists(os.path.join(drive_ckpt_dir, "latest_checkpoint.pth")):
            resume_path = os.path.join(drive_ckpt_dir, "latest_checkpoint.pth")

        start_epoch, best_val_dice = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

    # Training history
    train_losses = []
    val_losses = []
    val_dices = []

    # Load existing history if resuming
    history_path = os.path.join(ckpt_dir, "training_log.csv")
    if resume and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_losses.append(float(row['train_loss']))
                val_losses.append(float(row['val_loss']))
                val_dices.append(float(row['val_dice']))

    # Training loop
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"TRAINING START — {fold_str}")
    print(f"  Epochs: {start_epoch + 1} → {epochs}")
    print(f"  LR: {lr} (Paper: 6e-4, see config.py ambiguity note)")
    print(f"  Batch size: {batch_size}")
    print(f"  AMP: {USE_AMP}")
    print(f"  Loss: {loss}")
    print(f"  Scheduler: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | LR: {current_lr:.2e}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # Update scheduler (using val Dice as metric)
        scheduler.step(val_dice)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Best: {best_val_dice:.4f}")

        # Save latest checkpoint (for resume)
        latest_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_dice': best_val_dice,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'timestamp': run_timestamp,
            'fold': fold,
        }

        save_checkpoint(
            latest_state,
            os.path.join(ckpt_dir, "latest_checkpoint.pth"),
            os.path.join(drive_ckpt_dir, "latest_checkpoint.pth") if drive_ckpt_dir else None
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            latest_state['best_val_dice'] = best_val_dice

            save_checkpoint(
                latest_state,
                os.path.join(ckpt_dir, "best_model.pth"),
                os.path.join(drive_ckpt_dir, "best_model.pth") if drive_ckpt_dir else None
            )
            print(f"  → New best model saved! (Dice: {val_dice:.4f})")

        # Save training log
        with open(history_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'lr'])
            for ep, tl, vl, vd in zip(range(1, len(train_losses) + 1),
                                       train_losses, val_losses, val_dices):
                writer.writerow([ep, f"{tl:.6f}", f"{vl:.6f}", f"{vd:.6f}", f"{current_lr:.2e}"])

        # Copy log to Drive
        if drive_ckpt_dir:
            drive_history = os.path.join(drive_ckpt_dir, "training_log.csv")
            with open(drive_history, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'lr'])
                for ep, tl, vl, vd in zip(range(1, len(train_losses) + 1),
                                           train_losses, val_losses, val_dices):
                    writer.writerow([ep, f"{tl:.6f}", f"{vl:.6f}", f"{vd:.6f}", f"{current_lr:.2e}"])

    # Generate plots
    from src.utils import plot_metrics
    plots_dir = os.path.join(ckpt_dir, "plots")
    plot_metrics(train_losses, val_losses, val_dices, save_dir=plots_dir)
    if drive_ckpt_dir:
        plot_metrics(train_losses, val_losses, val_dices,
                     save_dir=os.path.join(drive_ckpt_dir, "plots"))

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {fold_str}")
    print(f"  Best Validation Dice: {best_val_dice:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    if drive_ckpt_dir:
        print(f"  Drive backup: {drive_ckpt_dir}")
    print(f"{'='*60}")

    return {
        'best_val_dice': best_val_dice,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_dices': val_dices,
        'fold': fold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Attention U-Net for Pancreas Segmentation"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory with preprocessed images/ and labels/")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Local checkpoint directory")
    parser.add_argument("--drive_checkpoint_dir", type=str, default=None,
                        help="Google Drive checkpoint directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Total training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start training from scratch (don't resume)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--fold", type=int, default=None,
                        help="Cross-validation fold number")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--loss", type=str, default='dicefocal',
                        choices=['dicefocal', 'dicebce', 'dice'],
                        help="Loss function (default: dicefocal)")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Focal gamma override (dicefocal only)")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Focal alpha override (dicefocal only)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        drive_checkpoint_dir=args.drive_checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
        num_workers=args.num_workers,
        fold=args.fold,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )


if __name__ == "__main__":
    main()
