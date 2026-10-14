# REPRODUCE

All commands from the repository root. Numbers in reports come only from the files these commands write.

## 1. Local: data, alignment gate, preprocessing, tests
```bash
python tools/fetch_drive.py --out data --subset images,labels,raw_labels      # course NIfTI (slow: ~0.35 MB/s)
python tools/check_alignment.py --images data/images --labels data/labels \
       --dicom data/dicom --raw_labels data/raw_labels --out results/alignment   # T1 (DICOM optional)
python -m src.data.prepare_data --images data/images --labels data/labels --out prep --workers 4   # T2
python tools/unpack_npz.py --src prep --dst prep
python -m src.test_all
```

## 2. Kaggle (2x T4 16 GB) - what the team actually runs
1. `python scripts/package.py` -> upload `dist/pancreas-code.zip` as Kaggle Dataset **pancreas-code**.
2. New notebook from `notebooks/kaggle_ablation.ipynb`; add the dataset; Accelerator GPU T4 x2; Internet On.
3. Interactive: cells 1-5 = tests, IDC download + T1 on all 80 + T2 (gates: identity rule, air <= 0.01, QC 80/80,
   splits == `splits/patient_splits.json`), benchmark (`results/benchmark.json`: N, K, schedule), T6 sanity.
4. Set `RUN_FULL = True`, `RUN_SANITY = False` -> Save & Run All (Commit). Re-running resumes from `last.pth`.
5. Download `/kaggle/working/results`.

Manual equivalents:
```bash
python tools/benchmark_steps.py --data /tmp/prep --wall_hours 3 --out results/benchmark.json
python scripts/kaggle_launch.py --data /tmp/prep --out results            # R0-R3, N/K/schedule from benchmark
python -m src.train --run_name R3_final --patch --norm group --deepsup --steps N --val_every K --data /tmp/prep --out results/R3_final
python -m src.evaluate --run_dir results/R3_final --data /tmp/prep       # ONLY script that reads test patients
python -m src.clinical --results results --data /tmp/prep                # C1 + C2 + case cards
python tools/make_figures.py --results results --data /tmp/prep          # all rubric figures/tables
```
OOM ladder (all patch runs identically): `--patch_size 112 112 80` -> `96 96 64` -> `--batch_size 1`.
