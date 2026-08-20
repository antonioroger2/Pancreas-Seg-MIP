import os
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
    RandCropByPosNegLabeld,
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


class Pancreas2DDataset(Dataset):
    """
    2D Slice-wise Dataset for Pancreas Segmentation in CT volumes.
    Extracts axial 2D slices containing image and binary mask.
    """
    def __init__(self, image_paths, mask_paths, transform=None, min_hu=-125.0, max_hu=225.0):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.slices_index = []

        # Index valid axial slices across 3D volumes
        for idx, (img_p, mask_p) in enumerate(zip(self.image_paths, self.mask_paths)):
            mask_obj = nib.load(mask_p)
            mask_data = mask_obj.get_fdata()
            # Find axial slices containing pancreas or non-empty region
            num_slices = mask_data.shape[2]
            for s in range(num_slices):
                # Optionally filter for slices with pancreas annotations
                if np.sum(mask_data[:, :, s]) > 0 or np.random.rand() < 0.2:
                    self.slices_index.append((idx, s))

    def __len__(self):
        return len(self.slices_index)

    def __getitem__(self, index):
        vol_idx, slice_idx = self.slices_index[index]
        
        img_vol = nib.load(self.image_paths[vol_idx]).get_fdata()
        mask_vol = nib.load(self.mask_paths[vol_idx]).get_fdata()

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
    Returns MONAI 3D/2D transformation pipelines for training and validation.
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


def create_dataloaders(data_dir, batch_size=4, val_split=0.2, num_workers=2):
    """
    Helper function to discover NIfTI images/labels in data_dir and return PyTorch DataLoaders.
    Expected folder structure:
        data_dir/
            images/ (e.g. image0001.nii.gz)
            labels/ (e.g. label0001.nii.gz)
    """
    image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii*")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii*")))

    assert len(image_paths) == len(mask_paths) and len(image_paths) > 0, \
        f"No matching image and label files found in {data_dir}"

    num_val = int(len(image_paths) * val_split)
    train_imgs, val_imgs = image_paths[num_val:], image_paths[:num_val]
    train_masks, val_masks = mask_paths[num_val:], mask_paths[:num_val]

    train_dataset = Pancreas2DDataset(train_imgs, train_masks)
    val_dataset = Pancreas2DDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
