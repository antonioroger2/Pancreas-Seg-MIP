import os
import argparse
from datetime import datetime
import csv
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import create_dataloaders
from model import build_model, DiceCELoss
from utils import set_seed, calculate_dice_score, plot_prediction_overlay, plot_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    use_cuda = (device.type == "cuda")

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    use_cuda = (device.type == "cuda")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs)
            for p, l in zip(preds, labels):
                score = calculate_dice_score(p, l)
                dice_scores.append(score)


    val_loss = running_loss / len(dataloader.dataset)
    val_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return val_loss, val_dice


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Pancreas Segmentation")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing images/ and labels/")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_drive_path", type=str, default=None, help="Google Drive path for backup checkpoints")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--light", action="store_true", default=True, help="Use lightweight U-Net for fast ~30 min training on Colab T4")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs (Multi-GPU CUDA DataParallel mode enabled)!")
    elif num_gpus == 1:
        print(f"Using Single GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU mode.")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_drive_path:
        os.makedirs(args.save_drive_path, exist_ok=True)

    # 1. Create DataLoaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    # 2. Build Model & Loss
    model = build_model("2d", in_channels=1, out_channels=1, light=args.light)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    criterion = DiceCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=(device.type == "cuda"))


    best_val_dice = 0.0
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"best_model_{run_timestamp}.pth"

    train_losses = []
    val_losses = []
    val_dices = []

    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Checkpoint Saving
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "timestamp": run_timestamp,
            }
            local_save = os.path.join(args.output_dir, timestamped_filename)
            latest_local_save = os.path.join(args.output_dir, "best_model.pth")
            torch.save(best_ckpt, local_save)
            torch.save(best_ckpt, latest_local_save)
            print(f" --> Saved new best model (Val Dice: {val_dice:.4f}) to {local_save}")

            if args.save_drive_path:
                drive_save = os.path.join(args.save_drive_path, timestamped_filename)
                latest_drive_save = os.path.join(args.save_drive_path, "best_model.pth")
                torch.save(best_ckpt, drive_save)
                torch.save(best_ckpt, latest_drive_save)
                print(f" --> Backed up best model to Google Drive: {drive_save}")

    print(f"\nTraining Complete! Best Validation Dice Score: {best_val_dice:.4f}")

    # 3. Save Training Metrics CSV
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for ep, tl, vl, vd in zip(range(1, args.epochs + 1), train_losses, val_losses, val_dices):
            writer.writerow([ep, tl, vl, vd])
    print(f"Saved training log to: {csv_path}")

    if args.save_drive_path:
        drive_csv_path = os.path.join(args.save_drive_path, "training_log.csv")
        with open(drive_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
            for ep, tl, vl, vd in zip(range(1, args.epochs + 1), train_losses, val_losses, val_dices):
                writer.writerow([ep, tl, vl, vd])
        print(f"Backed up training log to Google Drive: {drive_csv_path}")

    # 4. Generate & Save Publication Plots
    plots_dir = os.path.join(args.output_dir, "plots")
    plot_metrics(train_losses, val_losses, val_dices, save_dir=plots_dir)

    if args.save_drive_path:
        drive_plots_dir = os.path.join(args.save_drive_path, "plots")
        plot_metrics(train_losses, val_losses, val_dices, save_dir=drive_plots_dir)


if __name__ == "__main__":
    main()

