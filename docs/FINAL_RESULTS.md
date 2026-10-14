# Final results (R0-R4, C1, C2)

**Status: PENDING the Kaggle runs.** This file contains no numbers yet on purpose. After the runs, regenerate
`results/figures/final_results_auto.md` with `python tools/make_figures.py --results results --data /tmp/prep`
and copy its tables here verbatim (never retype or round by hand).

## Verdict rule (fixed before any result was seen)
Delta mean test Dice >= +0.02 -> improved; <= -0.02 -> worse; otherwise "no clear difference". Paired Wilcoxon p on
per-patient Dice is reported alongside. Caveats that apply to every line: n = 16 test patients, one CV fold (fold 0:
51 train / 13 val), one seed (42), one step budget N for all runs (R0 sees one 224x224x128 crop per step, R1-R4 two
128x128x96 patches per step).

| Addition | Comparison | Verdict | Numbers source |
|---|---|---|---|
| M1 patches + sliding window | R0 -> R1 | pending | results/figures/final_results_auto.md |
| M2 GroupNorm | R1 -> R2 | pending | idem |
| M3 deep supervision | R2 -> R3 | pending | idem |
| C1 volumetry | all runs | pending | results/clinical/volumetry_summary.json |
| C2 TTA uncertainty flag | R3 | pending | results/clinical/c2_summary.json |

## Paper vs ours
`results/figures/a_paper_vs_ours.md` (paper Dice 80.8 +- 2.1 %, HD95 8.1 +- 1.6 mm, ASSD 2.1 +- 0.3 mm, Jaccard n/r;
the paper's SD is most likely across folds, ours across 16 test patients - not like-for-like).

## Known deviations from the paper (details in CHANGELOG.md)
D1 focal without alpha (paper-exact) · D2 simple attention gate in the ablation (`--ag paper` implemented, optional) ·
D3 scheduler on validation Dice · D4 SD footnote · D5 80 cases (paper: 82/81/79) · D6 in-plane spacing 0.66-0.96 mm (13 cases checked locally, prep/meta)
in our data, not 0.5 mm · D7 one GPU augmentation for all runs · D8 full-volume 1 mm evaluation for all runs ·
D9 hardware is not the explanation (paper used a T4) · D10 4-flip TTA. NIH Pancreas-CT is healthy pancreas: C1/C2
measure segmentation/volume accuracy only, no disease claim.
