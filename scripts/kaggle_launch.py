"""Launch the R0-R4 ablation (T7) or the T6 sanity run: ONE independent process per GPU (CUDA_VISIBLE_DEVICES),
each running its runs sequentially (train -> test evaluation). No DDP. Re-running resumes from last.pth.
Works on any Linux GPU box (Kaggle 2x T4, a single RTX 6000 Ada, cloud runners): the GPU count is detected and the
schedule comes from benchmark.json (made with the same GPU count). Fails loudly if a GPU is missing (no CPU fallback).

  python scripts/kaggle_launch.py --data /tmp/prep --bench results/benchmark.json --out /kaggle/working/results
  python scripts/kaggle_launch.py --data /tmp/prep --sanity_steps 900 --K 150 --out /kaggle/working/sanity
Every run gets the same N and K (fair comparison); schedule = benchmark recommendation unless --gpu0/--gpu1 given.
"""
import argparse, json, os, subprocess, sys, threading

RUN_FLAGS = {"R0_baseline": "",
             "R1_patch": "--patch",
             "R2_gn": "--patch --norm group",
             "R3_final": "--patch --norm group --deepsup",
             "R4_noattn": "--patch --norm group --deepsup --no_attention"}


def chain(gpu, runs, a, N, K, extra=""):
    """Thread that runs this GPU's queue sequentially (train -> evaluate per run); returns the thread."""
    os.makedirs(a.logs, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), REQUIRE_CUDA="1")
    cmds = []
    for r in runs:
        out = os.path.join(a.out, r)
        cmds.append((r, [sys.executable, "-m", "src.train", "--run_name", r, *RUN_FLAGS[r].split(), *extra.split(),
                         "--steps", str(N), "--val_every", str(K), "--data", a.data, "--splits", a.splits, "--out", out]))
        if not a.no_eval:
            cmds.append((r, [sys.executable, "-m", "src.evaluate", "--run_dir", out, "--data", a.data, "--splits", a.splits]))

    def work():
        for r, c in cmds:
            with open(os.path.join(a.logs, f"{r}.log"), "a") as log:
                code = subprocess.call(c, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(f"GPU{gpu} {r} {c[2]} exit {code}", flush=True)
            if code:
                codes[gpu] = code
                return
    print(f"GPU{gpu}: {runs}")
    t = threading.Thread(target=work); t.start()
    return t


codes = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--splits", default="splits/patient_splits.json")
    ap.add_argument("--out", default="results"); ap.add_argument("--logs", default="results/logs")
    ap.add_argument("--bench", default="results/benchmark.json")
    ap.add_argument("--N", type=int); ap.add_argument("--K", type=int)
    ap.add_argument("--gpu0", default=""); ap.add_argument("--gpu1", default="")
    ap.add_argument("--batch_size", type=int, help="OOM ladder last rung: applied to ALL patch runs")
    ap.add_argument("--with_r4", action="store_true", help="append R4 to the shorter GPU queue")
    ap.add_argument("--patch_size", type=int, nargs=3, help="OOM ladder: applied to ALL patch runs")
    ap.add_argument("--sanity_steps", type=int, default=0, help="T6: short R0 + R3 (parallel on 2 GPUs, else sequential), no test eval")
    ap.add_argument("--no_eval", action="store_true")
    a = ap.parse_args()
    rec = json.load(open(a.bench))["recommendation"] if os.path.exists(a.bench) else {}
    N, K = a.N or rec.get("N"), a.K or rec.get("K")
    import torch
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "no CUDA GPU visible: refusing to train on CPU"
    extra = (f"--patch_size {' '.join(map(str, a.patch_size))} " if a.patch_size else "") +             (f"--batch_size {a.batch_size}" if a.batch_size else "")
    if a.sanity_steps:
        a.no_eval, N = True, a.sanity_steps                 # sanity never touches the test set
        queues = [["R0_baseline"], ["R3_final"]] if n_gpu >= 2 else [["R0_baseline", "R3_final"]]
    else:
        sched = rec.get("schedule", {})
        queues = [q.split(",") for q in (a.gpu0, a.gpu1) if q] or [sched[k] for k in sorted(sched) if sched[k]]
        assert queues, "no schedule: run tools/benchmark_steps.py first or pass --gpu0/--gpu1"
        assert len(queues) <= n_gpu, f"schedule needs {len(queues)} GPUs but {n_gpu} visible: re-run the benchmark here"
        if a.with_r4:
            m = rec.get("est_minutes_per_run", {})
            min(queues, key=lambda q: sum(m.get(r, 0) for r in q)).append("R4_noattn")
    assert N and K, "N and K must come from the benchmark (results/benchmark.json) or be given explicitly"
    print(f"N={N} K={K} GPUs={n_gpu} {extra}")
    for t in [chain(i, q, a, N, K, extra) for i, q in enumerate(queues)]:
        t.join()
    print("failed queues (gpu: exit code):", codes or "none")
    sys.exit(1 if codes else 0)


if __name__ == "__main__":
    main()
