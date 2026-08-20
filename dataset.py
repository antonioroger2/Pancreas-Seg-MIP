"""
dataset.py - Data loading pipeline for Pancreas CT Segmentation.

Supports two input modes:
  1. Pre-converted NIfTI volumes (recommended, use prepare_data.py first):
       data_dir/images/*.nii.gz  +  data_dir/labels/*.nii.gz

  2. Raw DICOM directories (auto-converts on-the-fly via SimpleITK):
       data_dir/images/PANCREAS_0001/StudyUID/SeriesUID/*.dcm
       data_dir/labels/*.nii.gz   (annotations are always NIfTI)
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import SimpleITK as sitk

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandRotated,
    RandFlipd,
    EnsureTyped,
)


def clip_and_normalize_hu(image_array, min_hu=-125.0, max_hu=225.0):
    """
    Clips CT Hounsfield Units (HU) to pancreatic soft tissue range [-125, 225]
    and normalizes pixel values to [0, 1].
    """
    clipped = np.clip(image_array, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    return normalized.astype(np.float32)


def load_volume(path):
    """
    Loads a 3D volume from either a NIfTI file or a DICOM series directory.
    Returns a numpy array of shape (H, W, D).
    """
    if os.path.isdir(path):
        # DICOM series directory
        return _load_dicom_volume(path)
    elif path.endswith((".nii", ".nii.gz")):
        # NIfTI file
        return nib.load(path).get_fdata().astype(np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def _load_dicom_volume(dicom_dir):
    """
    Reads a DICOM series from a directory (handles nested TCIA structure).
    Returns a numpy array of shape (H, W, D).
    """
    # Walk to find the deepest folder containing .dcm files
    series_dir = _find_dicom_series_dir(dicom_dir)
    if series_dir is None:
        raise FileNotFoundError(f"No DICOM series found in: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(series_dir)

    if len(dicom_filenames) == 0:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    reader.SetFileNames(dicom_filenames)
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    # Convert SimpleITK image to numpy (returns shape: D, H, W)
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    # Transpose to (H, W, D) to match nibabel convention
    array = np.transpose(array, (1, 2, 0))
    return array


def _find_dicom_series_dir(root_dir):
    """
    Walk a directory tree to find the folder that actually contains .dcm files.
    TCIA nests as: PatientID / StudyInstanceUID / SeriesInstanceUID / *.dcm
    """
    # Check if root_dir itself contains .dcm files
    dcm_files = glob.glob(os.path.join(root_dir, "*.dcm")) + \
                glob.glob(os.path.join(root_dir, "*.DCM"))
    if len(dcm_files) > 1:
        return root_dir

    # Walk subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dcm_count = sum(1 for f in filenames if f.lower().endswith(".dcm"))
        if dcm_count > 1:
            return dirpath
    return None


class Pancreas2DDataset(Dataset):
    """
    2D Slice-wise Dataset for Pancreas Segmentation in CT volumes.
    Extracts axial 2D slices from 3D volumes (NIfTI or DICOM).
    """
    def __init__(self, image_paths, mask_paths, transform=None, min_hu=-125.0, max_hu=225.0):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.slices_index = []

        # Index valid axial slices across 3D volumes
        print(f"  Indexing {len(self.image_paths)} volumes for 2D slices...")
        for idx, (img_p, mask_p) in enumerate(zip(self.image_paths, self.mask_paths)):
            mask_data = load_volume(mask_p) if not os.path.isdir(mask_p) else nib.load(mask_p).get_fdata()
            # Use nibabel for NIfTI masks (annotations are always NIfTI)
            if mask_p.endswith((".nii", ".nii.gz")):
                mask_data = nib.load(mask_p).get_fdata()
            else:
                mask_data = load_volume(mask_p)

            num_slices = mask_data.shape[2]
            for s in range(num_slices):
                has_pancreas = np.sum(mask_data[:, :, s]) > 0
                # Keep all pancreas slices + 20% of background slices for context
                if has_pancreas or np.random.rand() < 0.2:
                    self.slices_index.append((idx, s))

        print(f"  Indexed {len(self.slices_index)} total 2D slices.")

    def __len__(self):
        return len(self.slices_index)

    def __getitem__(self, index):
        vol_idx, slice_idx = self.slices_index[index]

        # Load volumes
        img_vol = load_volume(self.image_paths[vol_idx])
        mask_vol = load_volume(self.mask_paths[vol_idx])

        img_slice = img_vol[:, :, slice_idx]
        mask_slice = mask_vol[:, :, slice_idx]

        # Apply HU clipping and normalization
        img_slice = clip_and_normalize_hu(img_slice, self.min_hu, self.max_hu)
        mask_slice = (mask_slice > 0).astype(np.float32)

        # Convert to Tensor (Channel First: [1, H, W])
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0)

        data_dict = {"image": img_tensor, "label": mask_tensor}

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict["image"], data_dict["label"]


def get_monai_transforms(keys=["image", "label"], img_size=(256, 256)):
    """
    Returns MONAI transformation pipelines for training and validation.
    """
    train_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=225,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandRotated(keys=keys, range_x=0.3, prob=0.5, mode=["bilinear", "nearest"]),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        EnsureTyped(keys=keys),
    ])

    val_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=225,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=keys),
    ])

    return train_transforms, val_transforms


def discover_data_paths(data_dir):
    """
    Auto-discovers image and label file paths from data_dir.
    Supports both NIfTI files and DICOM patient directories.

    Expected layouts:
      Layout A (NIfTI - recommended):
        data_dir/images/PANCREAS_0001.nii.gz
        data_dir/labels/PANCREAS_0001.nii.gz

      Layout B (DICOM images + NIfTI labels):
        data_dir/images/PANCREAS_0001/   (nested DICOM dirs)
        data_dir/labels/label0001.nii.gz

    Returns:
        image_paths: sorted list of paths (files or directories)
        mask_paths:  sorted list of NIfTI label file paths
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # Try NIfTI images first
    nifti_images = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))

    if len(nifti_images) > 0:
        # Layout A: NIfTI images
        nifti_labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii*")))
        assert len(nifti_images) == len(nifti_labels) and len(nifti_images) > 0, \
            (f"NIfTI image/label count mismatch: {len(nifti_images)} images vs "
             f"{len(nifti_labels)} labels in {data_dir}. "
             f"Run prepare_data.py first to convert DICOM to NIfTI.")
        print(f"[Dataset] Found {len(nifti_images)} NIfTI image-label pairs.")
        return nifti_images, nifti_labels

    # Try DICOM patient directories
    dicom_patient_dirs = sorted([
        os.path.join(images_dir, d)
        for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d)) and not d.startswith(".")
    ]) if os.path.isdir(images_dir) else []

    if len(dicom_patient_dirs) > 0:
        print(f"[Dataset] Found {len(dicom_patient_dirs)} DICOM patient directories.")
        print("[Dataset] NOTE: For best performance, pre-convert with prepare_data.py")

        # Match each DICOM directory to its label file
        nifti_labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii*")))

        if len(nifti_labels) == 0:
            raise FileNotFoundError(
                f"No label NIfTI files found in {labels_dir}. "
                f"Download the TCIA annotations and place them in {labels_dir}."
            )

        # Build matched pairs by patient ID
        matched_images = []
        matched_labels = []

        for patient_dir in dicom_patient_dirs:
            patient_name = os.path.basename(patient_dir)
            match = re.search(r"(\d+)", patient_name)
            if match is None:
                continue
            patient_id = match.group(1).zfill(4)

            # Find matching label
            label_path = None
            for lp in nifti_labels:
                label_basename = os.path.basename(lp)
                if patient_id in label_basename:
                    label_path = lp
                    break

            if label_path:
                matched_images.append(patient_dir)
                matched_labels.append(label_path)

        assert len(matched_images) > 0, \
            (f"Could not match any DICOM patient folders to label files. "
             f"Check naming conventions in {images_dir} and {labels_dir}.")

        print(f"[Dataset] Matched {len(matched_images)} DICOM-label pairs.")
        return matched_images, matched_labels

    # Nothing found
    raise FileNotFoundError(
        f"No image data found in {images_dir}. Expected either:\n"
        f"  - NIfTI files: {images_dir}/*.nii.gz\n"
        f"  - DICOM dirs:  {images_dir}/PANCREAS_0001/\n"
        f"Run prepare_data.py to convert your TCIA DICOM download."
    )


def create_dataloaders(data_dir, batch_size=4, val_split=0.2, num_workers=2):
    """
    Discovers images/labels in data_dir and returns PyTorch DataLoaders.
    Supports both NIfTI files and DICOM directories automatically.

    Args:
        data_dir:    Root directory containing images/ and labels/ subdirectories.
        batch_size:  Batch size for DataLoader.
        val_split:   Fraction of data to use for validation (default 0.2 = 80/20 split).
        num_workers: Number of DataLoader workers.

    Returns:
        train_loader, val_loader
    """
    image_paths, mask_paths = discover_data_paths(data_dir)

    num_total = len(image_paths)
    if num_total == 1:
        # Sanity test mode with a single volume
        train_imgs, val_imgs = image_paths, image_paths
        train_masks, val_masks = mask_paths, mask_paths
        print(f"[Dataset] Sanity test mode: using 1 volume for both training and validation.")
    else:
        num_val = max(1, int(num_total * val_split))
        num_train = max(1, num_total - num_val)
        train_imgs, val_imgs = image_paths[num_val:], image_paths[:num_val]
        train_masks, val_masks = mask_paths[num_val:], mask_paths[:num_val]
        print(f"[Dataset] Train: {num_train} volumes | Val: {num_val} volumes")

    train_dataset = Pancreas2DDataset(train_imgs, train_masks)
    val_dataset = Pancreas2DDataset(val_imgs, val_masks)


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
