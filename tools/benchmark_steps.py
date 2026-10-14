"""T5 benchmark (run on the Kaggle T4): seconds/step + peak VRAM for R0-R4, one validation pass and one full-volume
test inference; proposes the shared step budget N, validation interval K and a 2-GPU schedule.

Rule: every run gets the SAME N (multiple of 100). K is set so validation costs ~15 % of the slowest run.
N = the largest value for which R0-R3 (longest-first on 2 GPUs, one process per GPU) finish within --wall_hours.
  python tools/benchmark_steps.py --data /tmp/prep --wall_hours 3 --out results/benchmark.json
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.augment import gpu_augment                                             # noqa: E402
from src.config import CROP_SIZE, PATCH_SIZE, SW_OVERLAP_TEST, SW_OVERLAP_VAL      # noqa: E402
from src.data.dataset import CropSampler, PatchSampler, load_case               # noqa: E402
from src.inference import predict_logits                                        # noqa: E402
from src.losses import DeepSupervisionLoss, DiceFocalLoss                       # noqa: E402
from src.models.model import build_model                                        # noqa: E402

RUNS = {"R0_baseline": dict(), "R1_patch": dict(patch=True), "R2_gn": dict(patch=True, norm="group"),
        "R3_final": dict(patch=True, norm="group", deepsup=True),
        "R4_noattn": dict(patch=True, norm="group", deepsup=True, attention=False)}


def bench(name, kw, data, train_ids, val_id, patch, iters, workers):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    is_patch = kw.get("patch", False)
    model = build_model(**kw).cuda().train()
    crit = DeepSupervisionLoss(DiceFocalLoss()).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler("cuda")
    bs = 2 if is_patch else 1
    ds = (PatchSampler(data, train_ids, (iters + 5) * bs, size=patch) if is_patch
          else CropSampler(data, train_ids, (iters + 5) * bs))
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True)
    t = None
    for i, (x, y) in enumerate(dl):
        if i == 5:
            torch.cuda.synchronize(); t = time.time()
        x, y = gpu_augment(x.cuda(non_blocking=True), y.cuda(non_blocking=True))
        with torch.autocast("cuda"):
            out = model(x)
        loss = crit(out, y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    s_step = (time.time() - t) / iters
    vram = torch.cuda.max_memory_allocated() / 1e9
    model.eval()
    img, _ = load_case(data, val_id)
    mode = "patch" if is_patch else "crop"
    res = {}
    for tag, ov in (("val_s_per_case", SW_OVERLAP_VAL), ("test_s_per_case", SW_OVERLAP_TEST)):
        torch.cuda.synchronize(); t = time.time()
        predict_logits(model, img, mode, patch, ov, CROP_SIZE, "cuda")
        torch.cuda.synchronize(); res[tag] = time.time() - t
    out = dict(run=name, s_per_step=s_step, peak_vram_gb=vram, patch=list(patch) if is_patch else list(CROP_SIZE),
               batch=bs, val_case_shape=list(img.shape), **res)
    print(json.dumps(out), flush=True)
    del model, opt
    return out


def main():
    torch.backends.cudnn.benchmark = True        # as in src/train.py
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--splits", default="splits/patient_splits.json")
    ap.add_argument("--iters", type=int, default=20); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--wall_hours", type=float, default=3.0, help="wall-clock budget for R0-R3 on 2 GPUs")
    ap.add_argument("--patch_size", type=int, nargs=3, default=list(PATCH_SIZE))
    ap.add_argument("--runs", default="R0_baseline,R1_patch,R2_gn,R3_final,R4_noattn")
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()
    sp = json.load(open(a.splits))["folds"][0]
    n_val = len(sp["val"])
    rows = []
    for name in a.runs.split(","):
        try:
            rows.append(bench(name, RUNS[name], a.data, sp["train"], sp["val"][0], tuple(a.patch_size), a.iters, a.workers))
        except torch.cuda.OutOfMemoryError:
            print(f"{name}: OOM at patch {a.patch_size} -> apply the OOM ladder to ALL patch runs "
                  f"(112 112 80 -> 96 96 64 -> --batch_size 1) and re-run", flush=True)
            rows.append(dict(run=name, oom=True)); torch.cuda.empty_cache()
    ok = {r["run"]: r for r in rows if not r.get("oom")}
    core = [r for r in ("R0_baseline", "R1_patch", "R2_gn", "R3_final") if r in ok]

    def plan(N):
        v_max = max(ok[r]["val_s_per_case"] for r in core) * n_val
        s_max = max(ok[r]["s_per_step"] for r in core)
        n_checks = max(4, int(0.15 * N * s_max / 0.85 / v_max))           # validation ~15 % of the slowest run
        K = max(50, int(math.ceil(N / n_checks / 50.0)) * 50)
        mins = {r: (N * ok[r]["s_per_step"] + math.ceil(N / K) * ok[r]["val_s_per_case"] * n_val
                    + ok[r]["test_s_per_case"] * 16) / 60 for r in ok}
        gpus = [[], []]
        for r in sorted(core, key=lambda r: -mins[r]):                      # longest-first onto the freer GPU
            min(gpus, key=lambda g: sum(mins[x] for x in g)).append(r)
        return K, mins, gpus, max(sum(mins[x] for x in g) for g in gpus)

    N = 6000
    while N > 100 and plan(N)[3] > a.wall_hours * 60:
        N -= 100
    K, mins, gpus, span = plan(N)
    rec = dict(N=N, K=K, n_val_checks=math.ceil(N / K), wall_hours=a.wall_hours, makespan_min=round(span, 1),
               est_minutes_per_run={r: round(m, 1) for r, m in mins.items()},
               schedule={"GPU0": gpus[0], "GPU1": gpus[1]}, n_val=n_val, patch_runs_ge_3000=N >= 3000,
               hours_needed_for_N3000=round(plan(3000)[3] / 60, 2), gpu=torch.cuda.get_device_name(0))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(runs=rows, recommendation=rec), open(a.out, "w"), indent=1)
    print("RECOMMENDATION", json.dumps(rec, indent=1))
    if N < 3000:
        print(f"WARNING: N={N} < 3000 within {a.wall_hours} h. N=3000 needs ~{rec['hours_needed_for_N3000']} h wall-clock. "
              f"Choose: raise --wall_hours, or accept N={N} and REPORT it (R0 is the bottleneck: one 224x224x128 crop/step).")

if __name__ == "__main__":
    main()
