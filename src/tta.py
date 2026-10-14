"""
tta.py - Flip test-time augmentation and voxel uncertainty (C2, inference only; decision D10: 4 flips).

FLIPS = identity + one flip per spatial axis. Each flipped volume is predicted with the run's own inference mode,
the probability is un-flipped and averaged; voxel entropy H(p) = -p log p - (1-p) log(1-p) of the mean probability;
case score = mean entropy inside the TTA mask (nan if the mask is empty).
"""
import numpy as np
import torch

from src.config import SW_OVERLAP_VAL
from src.inference import predict_logits

FLIPS = [(), (0,), (1,), (2,)]          # axes of the (H, W, D) volume


@torch.no_grad()
def tta_predict(model, image, cfg, overlap=SW_OVERLAP_VAL, device="cuda", flips=FLIPS):
    x = torch.as_tensor(np.asarray(image, dtype=np.float32), device=device)
    m = torch.zeros_like(x)
    for f in flips:
        xf = torch.flip(x, f) if f else x
        p = torch.sigmoid(predict_logits(model, xf, cfg["mode"], cfg["roi"], overlap, cfg["crop"], device))
        m += torch.flip(p, f) if f else p
    m /= len(flips)
    ent = -(m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8))
    mask = m > 0.5
    score = ent[mask].mean().item() if mask.any() else float("nan")
    return mask.cpu().numpy(), ent.cpu().numpy().astype(np.float16), score
