"""Launch the R0-R4 ablation (T7) or the T6 sanity run: ONE independent process per GPU (CUDA_VISIBLE_DEVICES),
each running its runs sequentially (train -> test evaluation). No DDP. Re-running resumes from last.pth.

  python scripts/kaggle_launch.py --data /tmp/prep --bench results/benchmark.json --out /kaggle/working/results
  python scripts/kaggle_launch.py --data /tmp/prep --sanity_steps 900 --K 150 --out /kaggle/working/sanity
Every run gets the same N and K (fair comparison); schedule = benchmark recommendation unless --gpu0/--gpu1 given.
"""
import argparse, json, os, subprocess, sys

RUN_FLAGS = {"R0_baseline": "",
             "R1_patch": "--patch",
             "R2_gn": "--patch --norm group",
             "R3_final": "--patch --norm group --deepsup",
             "R4_noattn": "--patch --norm group --deepsup --no_attention"}


def chain(gpu, runs, a, N, K, extra=""):
    os.makedirs(a.logs, exist_ok=True)
    cmds = []
    for r in runs:
        out = os.path.join(a.out, r)
        log = os.path.join(a.logs, f"{r}.log")
        cmds.append(f"{sys.executable} -m src.train --run_name {r} {RUN_FLAGS[r]} {extra} --steps {N} --val_every {K} "
                    f"--data {a.data} --splits {a.splits} --out {out} >> {log} 2>&1")
        if not a.no_eval:
            cmds.append(f"{sys.executable} -m src.evaluate --run_dir {out} --data {a.data} --splits {a.splits} >> {log} 2>&1")
    print(f"GPU{gpu}: {runs}")
    return subprocess.Popen(["bash", "-c", " ; ".join(cmds)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--splits", default="splits/patient_splits.json")
    ap.add_argument("--out", default="/kaggle/working/results"); ap.add_argument("--logs", default="/kaggle/working/logs")
    ap.add_argument("--bench", default="results/benchmark.json")
    ap.add_argument("--N", type=int); ap.add_argument("--K", type=int)
    ap.add_argument("--gpu0", default=""); ap.add_argument("--gpu1", default="")
    ap.add_argument("--with_r4", action="store_true", help="append R4 to the shorter GPU queue")
    ap.add_argument("--patch_size", type=int, nargs=3, help="OOM ladder: applied to ALL patch runs")
    ap.add_argument("--sanity_steps", type=int, default=0, help="T6: short R0 (GPU0) + R3 (GPU1), no test eval")
    ap.add_argument("--no_eval", action="store_true")
    a = ap.parse_args()
    rec = json.load(open(a.bench))["recommendation"] if os.path.exists(a.bench) else {}
    N, K = a.N or rec.get("N"), a.K or rec.get("K")
    extra = f"--patch_size {' '.join(map(str, a.patch_size))}" if a.patch_size else ""
    if a.sanity_steps:
        a.no_eval, N = True, a.sanity_steps                 # sanity never touches the test set
        q0, q1 = ["R0_baseline"], ["R3_final"]
    else:
        q0 = a.gpu0.split(",") if a.gpu0 else rec["schedule"]["GPU0"]
        q1 = a.gpu1.split(",") if a.gpu1 else rec["schedule"]["GPU1"]
        if a.with_r4:
            m = rec.get("est_minutes_per_run", {})
            (q0 if sum(m.get(r, 0) for r in q0) <= sum(m.get(r, 0) for r in q1) else q1).append("R4_noattn")
    assert N and K, "N and K must come from the benchmark (results/benchmark.json) or be given explicitly"
    print(f"N={N} K={K} {extra}")
    procs = [chain(0, q0, a, N, K, extra), chain(1, q1, a, N, K, extra)]
    codes = [p.wait() for p in procs]
    print("exit codes", codes)


if __name__ == "__main__":
    main()
