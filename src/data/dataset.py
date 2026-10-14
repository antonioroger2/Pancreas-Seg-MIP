"""
dataset.py - Training samplers over the cached full 1 mm volumes (T2 output).

Storage: <data>/npy/<PID>_image.npy (float16) + <PID>_label.npy (uint8), memory-mapped (tools/unpack_npz.py),
falling back to <data>/cases/<PID>.npz. Arrays are (H, W, D).

R0  CropSampler : the blind geometric-centre 224x224x128 crop of each training volume (no GT), batch 1.
R1+ PatchSampler: 128x128x96 patches, pos:neg = 1:2. Same rule as MONAI RandCropByPosNegLabeld
                  (centre on a random foreground voxel with prob pos/(pos+neg), else on a random background
                  voxel; centre clamped so the patch lies in the volume) but with cached foreground indices
                  instead of MONAI's per-call full-volume index arrays (~320 MB per call). GT is used ONLY to
                  choose training-patch locations.
Sample i is a pure function of (seed, i): reproducible across workers and exact on resume.
Training cases are visited in a fresh permutation every pass ("epoch" of the case list).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CROP_SIZE, PATCH_SIZE, POS_NEG
from src.data.preprocessing import blind_center_crop, crop_at


def load_case(data_dir: str, pid: str, mmap: bool = True):
    npy = os.path.join(data_dir, "npy", f"{pid}_image.npy")
    if os.path.exists(npy):
        mode = "r" if mmap else None
        return np.load(npy, mmap_mode=mode), np.load(npy.replace("_image", "_label"), mmap_mode=mode)
    z = np.load(os.path.join(data_dir, "cases", f"{pid}.npz"))
    return z["image"], z["label"]


class _Sampler(Dataset):
    def __init__(self, data_dir, ids, n_samples, seed=42, start=0):
        self.data_dir, self.ids, self.n, self.seed, self.start = data_dir, list(ids), n_samples, seed, start

    def __len__(self):
        return self.n - self.start

    def case_for(self, i):
        e, j = divmod(i, len(self.ids))
        return self.ids[np.random.default_rng((self.seed, 0, e)).permutation(len(self.ids))[j]]

    @staticmethod
    def to_tensor(img, lbl):
        return (torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(lbl, dtype=np.float32))[None])


class CropSampler(_Sampler):
    """R0: whole blind-centre crop of a training volume."""

    def __init__(self, *a, size=CROP_SIZE, **k):
        super().__init__(*a, **k); self.size = tuple(size)

    def __getitem__(self, idx):
        img, lbl = load_case(self.data_dir, self.case_for(idx + self.start))
        off = blind_center_crop(img.shape, self.size)
        return self.to_tensor(crop_at(img, off, self.size), crop_at(lbl, off, self.size))


class PatchSampler(_Sampler):
    """R1-R4: foreground-biased random patches."""

    def __init__(self, *a, size=PATCH_SIZE, pos_neg=POS_NEG, **k):
        super().__init__(*a, **k)
        self.size, self.p_pos = tuple(size), pos_neg[0] / (pos_neg[0] + pos_neg[1])
        self._fg = {}

    def fg_indices(self, pid, lbl):
        if pid not in self._fg:                      # per worker; ~60k voxels per case
            self._fg[pid] = np.argwhere(np.asarray(lbl) > 0).astype(np.int32)
        return self._fg[pid]

    def centre(self, rng, pid, lbl):
        if rng.random() < self.p_pos:
            fg = self.fg_indices(pid, lbl)
            return fg[rng.integers(len(fg))], True
        while True:                                  # background ~99.9 % of voxels: a few draws at most
            c = np.array([rng.integers(s) for s in lbl.shape])
            if lbl[tuple(c)] == 0:
                return c, False

    def __getitem__(self, idx):
        i = idx + self.start
        pid = self.case_for(i)
        img, lbl = load_case(self.data_dir, pid)
        rng = np.random.default_rng((self.seed, 1, i))
        c, _ = self.centre(rng, pid, lbl)
        # clamp so the patch lies inside the volume (volumes smaller than the patch are zero-padded)
        off = tuple(int(min(max(ci - t // 2, 0), max(s - t, 0))) if s >= t else -((t - s) // 2)
                    for ci, t, s in zip(c, self.size, img.shape))
        return self.to_tensor(crop_at(img, off, self.size), crop_at(lbl, off, self.size))
