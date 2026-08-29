"""
cross_validation.py - 5-Fold Cross-Validation with Independent Test Set.

Implements the paper's evaluation strategy (Section 2.4):
  - Hold out N_TEST_CASES patients for independent testing
  - 5-fold CV on remaining patients
  - Patient-level splitting (no data leakage)
  - Saves splits to JSON for reproducibility

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import os
import json
import argparse
import numpy as np

from config import (
    N_FOLDS, N_TEST_CASES, SEED, DEFAULT_DATA_DIR,
    DEFAULT_CHECKPOINT_DIR, DEFAULT_SPLITS_DIR,
    DEFAULT_EPOCHS, BATCH_SIZE, LEARNING_RATE,
)
from dataset import discover_data_paths
from train import train
from utils import set_seed


def create_patient_splits(n_patients: int, n_folds: int = N_FOLDS,
                           n_test: int = N_TEST_CASES,
                           seed: int = SEED,
                           splits_dir: str = DEFAULT_SPLITS_DIR) -> dict:
    """
    Create patient-level splits for cross-validation + independent test set.

    PAPER AMBIGUITY #3:
        Paper states 5 folds of 13 cases + 16 test cases = 81 ≠ 82.
        TCIA v2 has 80 cases. We adapt fold sizes to available data.

    Args:
        n_patients: total number of patients
        n_folds: number of CV folds (default 5)
        n_test: number of test cases to hold out
        seed: random seed for reproducibility
        splits_dir: directory to save split files

    Returns:
        dict with 'test_indices' and 'folds' (list of {train, val} indices)
    """
    set_seed(seed)

    # Shuffle patient indices
    all_indices = list(range(n_patients))
    np.random.shuffle(all_indices)

    # Hold out test set
    n_test_actual = min(n_test, n_patients // 4)  # Safety: don't hold out too many
    test_indices = sorted(all_indices[:n_test_actual])
    cv_indices = all_indices[n_test_actual:]

    # Create k folds from CV indices
    np.random.shuffle(cv_indices)
    fold_size = len(cv_indices) // n_folds
    folds = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            val_idx = sorted(cv_indices[start:])  # Last fold gets remainder
        else:
            val_idx = sorted(cv_indices[start:start + fold_size])

        train_idx = sorted([i for i in cv_indices if i not in val_idx])
        folds.append({
            'fold': fold_idx,
            'train_indices': train_idx,
            'val_indices': val_idx,
        })

    splits = {
        'n_patients': n_patients,
        'n_folds': n_folds,
        'n_test': len(test_indices),
        'seed': seed,
        'test_indices': test_indices,
        'folds': folds,
    }

    # Save splits
    os.makedirs(splits_dir, exist_ok=True)
    splits_file = os.path.join(splits_dir, "patient_splits.json")
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"[Splits] Saved to {splits_file}")

    # Also save per-fold files for convenience
    for fold_data in folds:
        fold_idx = fold_data['fold']
        fold_file = os.path.join(splits_dir, f"fold_{fold_idx}.json")
        with open(fold_file, 'w') as f:
            json.dump({
                'fold': fold_idx,
                'train_indices': fold_data['train_indices'],
                'val_indices': fold_data['val_indices'],
                'test_indices': test_indices,
            }, f, indent=2)

    # Summary
    print(f"\n[Splits] Patient-Level Split Summary:")
    print(f"  Total patients: {n_patients}")
    print(f"  Test set: {len(test_indices)} patients")
    print(f"  CV patients: {len(cv_indices)}")
    for fold_data in folds:
        print(f"  Fold {fold_data['fold']}: train={len(fold_data['train_indices'])}, "
              f"val={len(fold_data['val_indices'])}")

    return splits


def load_splits(splits_dir: str = DEFAULT_SPLITS_DIR) -> dict:
    """Load previously saved patient splits."""
    splits_file = os.path.join(splits_dir, "patient_splits.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"No splits file found at {splits_file}. "
            f"Run create_patient_splits() first."
        )

    with open(splits_file, 'r') as f:
        splits = json.load(f)

    print(f"[Splits] Loaded splits from {splits_file}")
    print(f"  Folds: {len(splits['folds'])}, Test: {splits['n_test']}")
    return splits


def run_cross_validation(data_dir: str = DEFAULT_DATA_DIR,
                          checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                          drive_checkpoint_dir: str = None,
                          splits_dir: str = DEFAULT_SPLITS_DIR,
                          fold: int = None,
                          epochs: int = DEFAULT_EPOCHS,
                          batch_size: int = BATCH_SIZE,
                          lr: float = LEARNING_RATE,
                          num_workers: int = 2,
                          loss: str = 'dicefocal',
                          focal_gamma: float = None,
                          focal_alpha: float = None):
    """
    Run 5-fold cross-validation (or a specific fold).

    Args:
        data_dir: preprocessed data directory
        checkpoint_dir: checkpoint save directory
        drive_checkpoint_dir: Google Drive backup directory
        splits_dir: directory containing split files
        fold: if specified, run only this fold (0-indexed)
        epochs: training epochs per fold
        batch_size: batch size
        lr: learning rate
        num_workers: DataLoader workers
    """
    # Discover data
    image_paths, mask_paths = discover_data_paths(data_dir)
    n_patients = len(image_paths)

    # Create or load splits
    splits_file = os.path.join(splits_dir, "patient_splits.json")
    if os.path.exists(splits_file):
        splits = load_splits(splits_dir)
        if splits['n_patients'] != n_patients:
            print(f"[WARNING] Split file has {splits['n_patients']} patients, "
                  f"but found {n_patients}. Regenerating splits.")
            splits = create_patient_splits(n_patients, splits_dir=splits_dir)
    else:
        splits = create_patient_splits(n_patients, splits_dir=splits_dir)

    # Determine which folds to run
    if fold is not None:
        folds_to_run = [f for f in splits['folds'] if f['fold'] == fold]
        if not folds_to_run:
            raise ValueError(f"Fold {fold} not found in splits")
    else:
        folds_to_run = splits['folds']

    # Run training for each fold
    all_results = []

    for fold_data in folds_to_run:
        fold_idx = fold_data['fold']
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold_idx}")
        print(f"# Train: {len(fold_data['train_indices'])} patients")
        print(f"# Val:   {len(fold_data['val_indices'])} patients")
        print(f"{'#'*60}")

        result = train(
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            drive_checkpoint_dir=drive_checkpoint_dir,
            fold=fold_idx,
            train_indices=fold_data['train_indices'],
            val_indices=fold_data['val_indices'],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            resume=True,
            num_workers=num_workers,
            loss=loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        all_results.append(result)

    # Summary across folds
    if len(all_results) > 1:
        dice_scores = [r['best_val_dice'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY ({len(all_results)} folds)")
        print(f"{'='*60}")
        for r in all_results:
            print(f"  Fold {r['fold']}: Best Dice = {r['best_val_dice']:.4f}")
        print(f"\n  Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"  Range: [{min(dice_scores):.4f}, {max(dice_scores):.4f}]")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="5-Fold Cross-Validation for 3D Pancreas Segmentation"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--drive_checkpoint_dir", type=str, default=None)
    parser.add_argument("--splits_dir", type=str, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run specific fold (0-indexed). Omit to run all.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--loss", type=str, default='dicefocal',
                        choices=['dicefocal', 'dicebce', 'dice'],
                        help="Loss function (default: dicefocal)")
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--focal_alpha", type=float, default=None)
    args = parser.parse_args()

    run_cross_validation(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        drive_checkpoint_dir=args.drive_checkpoint_dir,
        splits_dir=args.splits_dir,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )


if __name__ == "__main__":
    main()
