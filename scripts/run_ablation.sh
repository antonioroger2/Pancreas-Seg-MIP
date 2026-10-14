#!/usr/bin/env bash
# One-command reproduction of the ablation on a 2-GPU Linux box (Kaggle T4 x2 or similar).
# Usage: bash scripts/run_ablation.sh /tmp/prep results            (N, K, schedule from results/benchmark.json)
#        N=3000 K=250 bash scripts/run_ablation.sh /tmp/prep results (explicit budget)
set -euo pipefail
DATA=${1:?prep dir with npy/ or cases/}; OUT=${2:-results}
python -m src.test_all
[ -f results/benchmark.json ] || python tools/benchmark_steps.py --data "$DATA" --out results/benchmark.json
python scripts/kaggle_launch.py --data "$DATA" --out "$OUT" --logs "$OUT/logs" ${N:+--N $N} ${K:+--K $K}
python -m src.clinical --results "$OUT" --data "$DATA"
python tools/make_figures.py --results "$OUT"
