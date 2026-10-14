"""
inference.py - Full-volume prediction shared by validation, test evaluation and TTA (decision D8).

mode "crop"  (R0): blind-centre 224x224x128 crop -> model -> paste back; everything outside the crop is
                   background (logit BG_LOGIT).
mode "patch" (R1-R4): MONAI sliding_window_inference, ROI = training patch size, sw_batch 4, gaussian blending.
No ground truth is used. Returns full-size logits (H, W, D) as a float32 torch tensor on `device`.
"""
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.config import CROP_SIZE, PATCH_SIZE, SW_BATCH, SW_OVERLAP_TEST
from src.data.preprocessing import blind_center_crop

BG_LOGIT = -20.0      # sigmoid(-20) ~ 2e-9: "outside the R0 crop = background"


@torch.no_grad()
def predict_logits(model, image, mode="patch", roi=PATCH_SIZE, overlap=SW_OVERLAP_TEST, crop=CROP_SIZE,
                   device="cuda", amp=True):
    """image: (H, W, D) numpy/tensor in [0, 1]. Model must be in eval mode."""
    x = torch.as_tensor(np.asarray(image, dtype=np.float32) if not torch.is_tensor(image) else image,
                        dtype=torch.float32, device=device)[None, None]
    dev_type = torch.device(device).type
    with torch.autocast(dev_type, enabled=amp and dev_type == "cuda"):
        if mode == "crop":
            off = blind_center_crop(x.shape[2:], crop)
            pad = [(max(-o, 0), max(o + t - s, 0)) for o, t, s in zip(off, crop, x.shape[2:])]
            xp = torch.nn.functional.pad(x, [p for ax in reversed(pad) for p in ax])
            st = [max(o, 0) for o in off]
            out = model(xp[..., st[0]:st[0] + crop[0], st[1]:st[1] + crop[1], st[2]:st[2] + crop[2]]).float()
            full = torch.full(x.shape, BG_LOGIT, device=device)
            src = tuple(slice(max(-o, 0), max(-o, 0) + min(o + t, s) - max(o, 0)) for o, t, s in zip(off, crop, x.shape[2:]))
            dst = tuple(slice(max(o, 0), min(o + t, s)) for o, t, s in zip(off, crop, x.shape[2:]))
            full[(..., *dst)] = out[(..., *src)]
        else:
            full = sliding_window_inference(x, tuple(roi), SW_BATCH, model, overlap=overlap, mode="gaussian").float()
    return full[0, 0]
