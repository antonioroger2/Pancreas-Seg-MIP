"""
train.py - Step-based trainer for R0-R4 (one process per GPU; no DDP).

Defaults reproduce R0 (paper-faithful): BatchNorm, blind-centre 224x224x128 crop, batch 1, Dice+Focal (gamma 2,
no alpha), Adam 6e-4 wd 0, AMP, simple attention gate. Flags add M1 (--patch), M2 (--norm group), M3 (--deepsup).
Validation every K steps on the FULL 1 mm volume of the fold's validation patients (D8); ReduceLROnPlateau on
val Dice (D3, --sched_metric loss for the paper's choice) with patience = 1/3 of the number of validation checks.
best.pth = best val Dice; last.pth every --ckpt_every steps -> automatic resume. Test patients are never loaded.

  python -m src.train --run_name R0_baseline --data /tmp/prep --steps N --val_every K --out results/R0_baseline
"""
import argparse, csv, json, math, os, platform, subprocess, time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augment import gpu_augment
from src.config import (CROP_SIZE, FOCAL_GAMMA, LEARNING_RATE, PATCH_BATCH, PATCH_SIZE, POS_NEG, SCHEDULER_FACTOR,
                        SCHEDULER_MIN_LR, SEED, SW_OVERLAP_VAL, WEIGHT_DECAY, EVAL_THRESHOLD)
from src.data.dataset import CropSampler, PatchSampler, load_case
from src.inference import predict_logits
from src.losses import DeepSupervisionLoss, DiceFocalLoss
from src.models.model import build_model, count_parameters
from src.utils import set_seed

LOG_COLS = ["step", "train_loss", "val_dice", "val_loss", "lr", "seconds", "peak_vram_gb"]


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True); p.add_argument("--out", required=True)
    p.add_argument("--data", required=True); p.add_argument("--splits", default="splits/patient_splits.json")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--norm", choices=["batch", "group"], default="batch")
    p.add_argument("--patch", action="store_true")
    p.add_argument("--patch_size", type=int, nargs=3, default=list(PATCH_SIZE))
    p.add_argument("--batch_size", type=int, default=None, help="default 1 (R0) / 2 (--patch)")
    p.add_argument("--pos_neg", type=int, nargs=2, default=list(POS_NEG))
    p.add_argument("--deepsup", action="store_true"); p.add_argument("--no_attention", action="store_true")
    p.add_argument("--ag", choices=["simple", "paper"], default="simple")
    p.add_argument("--focal_alpha", type=float, default=None); p.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA)
    p.add_argument("--sched_metric", choices=["dice", "loss"], default="dice")
    p.add_argument("--steps", type=int, required=True); p.add_argument("--val_every", type=int, required=True)
    p.add_argument("--ckpt_every", type=int, default=300); p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--val_overlap", type=float, default=SW_OVERLAP_VAL)
    p.add_argument("--val_max_cases", type=int, default=0, help="0 = all validation patients")
    p.add_argument("--lr", type=float, default=LEARNING_RATE); p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--no_amp", action="store_true")
    a = p.parse_args(argv)
    a.batch_size = a.batch_size or (PATCH_BATCH if a.patch else 1)
    return a


def model_config(a):
    """Everything evaluate.py / tta.py need to rebuild the model and its inference mode."""
    return dict(norm=a.norm, deepsup=a.deepsup, attention=not a.no_attention, ag=a.ag,
                mode="patch" if a.patch else "crop", roi=list(a.patch_size), crop=list(CROP_SIZE))


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def validate(model, data, ids, cfg, crit, device, overlap, amp):
    """Mean Dice (threshold 0.5) and mean Dice+Focal loss over full validation volumes."""
    model.eval()
    dices, losses = [], []
    for pid in ids:
        img, lbl = load_case(data, pid)
        logits = predict_logits(model, img, cfg["mode"], cfg["roi"], overlap, cfg["crop"], device, amp)
        y = torch.from_numpy(np.array(lbl, dtype=np.float32)).to(device)
        p = logits > 0          # sigmoid > 0.5
        inter, tot = (p & (y > 0)).sum().item(), p.sum().item() + y.sum().item()
        dices.append(1.0 if tot == 0 else 2 * inter / tot)
        losses.append(crit(logits[None, None], y[None, None]).item())
    model.train()
    return float(np.mean(dices)), float(np.mean(losses))


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = not a.no_amp and device == "cuda"
    split = json.load(open(a.splits))
    fold = split["folds"][a.fold]
    train_ids, val_ids = fold["train"], fold["val"][:a.val_max_cases or None]
    assert not set(train_ids + val_ids) & set(split["test"]), "test patients must never be used in training"

    cfg = model_config(a)
    model = build_model(**cfg).to(device)
    crit_main = DiceFocalLoss(focal_gamma=a.focal_gamma, focal_alpha=a.focal_alpha)
    crit = DeepSupervisionLoss(crit_main).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=WEIGHT_DECAY)
    n_checks = math.ceil(a.steps / a.val_every)
    patience = max(1, n_checks // 3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max" if a.sched_metric == "dice" else "min",
                                                       factor=SCHEDULER_FACTOR, patience=patience, min_lr=SCHEDULER_MIN_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step, best, t_prev = 0, -1.0, 0.0
    last, log_path = os.path.join(a.out, "last.pth"), os.path.join(a.out, "train_log.csv")
    if os.path.exists(last):                                          # automatic resume
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); torch.set_rng_state(ck["rng_cpu"].cpu())
        if device == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        step, best, t_prev = ck["step"], ck["best"], ck["seconds"]
        rows = [r for r in csv.DictReader(open(log_path))] if os.path.exists(log_path) else []
        rows = [r for r in rows if int(r["step"]) <= step]
        print(f"[resume] step {step}, best val Dice {best:.4f}", flush=True)
    else:
        rows = []
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS); w.writeheader(); w.writerows(rows)

    config = dict(vars(a), **cfg, params=count_parameters(model)["total"], git=git_hash(), n_train=len(train_ids),
                  val_ids=val_ids, train_ids=train_ids, patience_checks=patience, n_val_checks=n_checks, amp=amp,
                  focal_alpha=a.focal_alpha, threshold=EVAL_THRESHOLD, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if device == "cuda" else platform.processor())
    json.dump(config, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"[{a.run_name}] {config['params']:,} params | {cfg} | batch {a.batch_size} | steps {a.steps} | "
          f"val every {a.val_every} on {len(val_ids)} cases | patience {patience} checks | {config['gpu']}", flush=True)

    Sampler = PatchSampler if a.patch else CropSampler
    kw = dict(size=a.patch_size, pos_neg=a.pos_neg) if a.patch else {}
    ds = Sampler(a.data, train_ids, a.steps * a.batch_size, seed=a.seed, start=step * a.batch_size, **kw)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=device == "cuda",
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, run_loss, n_loss = time.time() - t_prev, 0.0, 0

    def save_last():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), step=step, best=best, seconds=time.time() - t0,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state() if device == "cuda" else None), last + ".tmp")
        os.replace(last + ".tmp", last)

    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y = gpu_augment(x, y)
        with torch.autocast(device, enabled=amp):
            out = model(x)
        loss = crit(out, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1; run_loss += loss.item(); n_loss += 1
        row = None
        if step % a.val_every == 0 or step == a.steps:
            vd, vl = validate(model, a.data, val_ids, cfg, crit_main, device, a.val_overlap, amp)
            sched.step(vd if a.sched_metric == "dice" else vl)
            if vd > best:
                best = vd
                torch.save(dict(model=model.state_dict(), step=step, val_dice=vd, config=cfg), os.path.join(a.out, "best.pth"))
            row = dict(val_dice=f"{vd:.5f}", val_loss=f"{vl:.5f}")
        if row is not None or step % a.log_every == 0:
            row = dict(row or {}, step=step, train_loss=f"{run_loss / max(n_loss, 1):.5f}",
                       lr=f"{opt.param_groups[0]['lr']:.3e}", seconds=f"{time.time() - t0:.1f}",
                       peak_vram_gb=f"{torch.cuda.max_memory_allocated() / 1e9:.2f}" if device == "cuda" else "")
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_COLS).writerow(row)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            run_loss, n_loss = 0.0, 0
        if step % a.ckpt_every == 0 or step == a.steps:
            save_last()
    print(f"[{a.run_name}] done: {step} steps, best val Dice {best:.4f}, {time.time() - t0:.0f} s", flush=True)
    from src.utils import plot_train_log
    plot_train_log(log_path, os.path.join(a.out, "curves.png"), a.run_name)


if __name__ == "__main__":
    main()
