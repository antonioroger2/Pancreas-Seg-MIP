"""
dataset.py - 3D Volume Dataset for Pancreas CT Segmentation.

Features:
  - Lazy loading: volumes loaded from disk on-demand
  - Supports preprocessed NIfTI volumes
  - 3D augmentation (rotation, flip, shift) for training
  - Patient-level data organization

Returns tensors of shape [B, 1, H, W, D] = [B, 1, 224, 224, 128]

Reference:
  "Deep Learning Model with Attention Mechanism for a 3D Pancreas Segmentation in CT Scans"
  Mathematics 2025, 13, 3942
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

from config import (
    HU_MIN, HU_MAX, TARGET_SPACING, CROP_SIZE,
    AUG_ROTATION_RANGE, AUG_FLIP_PROB, AUG_SHIFT_RANGE,
    BATCH_SIZE,
)


class PancreasVolumeDataset(Dataset):
    """
    3D Volume Dataset -- loads one preprocessed volume per __getitem__ call.

    Each sample returns:
        image: torch.Tensor of shape (1, H, W, D)
        mask:  torch.Tensor of shape (1, H, W, D)
    """

    def __init__(self, image_paths: list, mask_paths: list,
                 augment: bool = False, preprocessed: bool = True):
        assert len(image_paths) == len(mask_paths), \
            f"Image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}"

        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.augment = augment
        self.preprocessed = preprocessed

        for p in self.image_paths + self.mask_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_nii = nib.load(self.image_paths[index])
        mask_nii = nib.load(self.mask_paths[index])

        image = img_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        if not self.preprocessed:
            from preprocessing import normalize_to_01, center_crop_or_pad
            image = normalize_to_01(image, HU_MIN, HU_MAX)
            mask = (mask > 0).astype(np.float32)
            image = center_crop_or_pad(image, CROP_SIZE)
            mask = center_crop_or_pad(mask, CROP_SIZE)
        else:
            mask = (mask > 0).astype(np.float32)

        if self.augment:
            image, mask = self._augment_3d(image, mask)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor

    def _augment_3d(self, image: np.ndarray, mask: np.ndarray):
        """
        Apply 3D augmentations per paper Section 2.2:
            - Random rotation +/-10 degrees per axis
            - Random flip, prob 0.5
            - Random shift up to 10% per axis
        """
        from scipy.ndimage import rotate, shift

        # Random rotation
        for axis_pair in [(0, 1), (0, 2), (1, 2)]:
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-AUG_ROTATION_RANGE, AUG_ROTATION_RANGE)
                image = rotate(image, angle, axes=axis_pair, reshape=False,
                             order=1, mode='nearest')
                mask = rotate(mask, angle, axes=axis_pair, reshape=False,
                            order=0, mode='nearest')

        # Random flip
        for axis in range(3):
            if np.random.rand() < AUG_FLIP_PROB:
                image = np.flip(image, axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()

        # Random shift
        if np.random.rand() < 0.5:
            max_shifts = [int(s * AUG_SHIFT_RANGE) for s in image.shape]
            shifts = [np.random.randint(-ms, ms + 1) if ms > 0 else 0 for ms in max_shifts]
            image = shift(image, shifts, order=1, mode='nearest')
            mask = shift(mask, shifts, order=0, mode='nearest')

        mask = (mask > 0.5).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return image, mask

    def get_patient_id(self, index: int) -> str:
        """Extract patient ID from filename."""
        basename = os.path.basename(self.image_paths[index])
        match = re.search(r'(\d+)', basename)
        return match.group(1).zfill(4) if match else basename


def discover_data_paths(data_dir: str):
    """Auto-discovers image and label file paths from data_dir."""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    mask_paths = sorted(glob.glob(os.path.join(labels_dir, "*.nii*")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No NIfTI images found in {images_dir}")
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(image_paths)} images vs "
            f"{len(mask_paths)} masks in {data_dir}"
        )

    print(f"[Dataset] Found {len(image_paths)} matched image-mask pairs in {data_dir}")
    return image_paths, mask_paths


def create_fold_dataloaders(data_dir: str, train_indices: list, val_indices: list,
                            batch_size: int = BATCH_SIZE,
                            num_workers: int = 2,
                            preprocessed: bool = True):
    """Create train and validation DataLoaders for a specific CV fold."""
    image_paths, mask_paths = discover_data_paths(data_dir)

    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]

    train_dataset = PancreasVolumeDataset(
        train_images, train_masks, augment=True, preprocessed=preprocessed
    )
    val_dataset = PancreasVolumeDataset(
        val_images, val_masks, augment=False, preprocessed=preprocessed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[Dataset] Train: {len(train_dataset)} volumes | Val: {len(val_dataset)} volumes")
    return train_loader, val_loader


def create_simple_dataloaders(data_dir: str, val_split: float = 0.2,
                              batch_size: int = BATCH_SIZE,
                              num_workers: int = 2,
                              preprocessed: bool = True):
    """Create simple train/val DataLoaders with a percentage split."""
    image_paths, mask_paths = discover_data_paths(data_dir)

    n = len(image_paths)
    n_val = max(1, int(n * val_split))

    val_indices = list(range(n_val))
    train_indices = list(range(n_val, n))

    if n == 1:
        train_indices = [0]
        val_indices = [0]

    return create_fold_dataloaders(
        data_dir, train_indices, val_indices,
        batch_size, num_workers, preprocessed
    )
