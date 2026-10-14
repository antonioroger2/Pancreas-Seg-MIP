#!/bin/sh
# nnU-Net v2 ResEnc-M (3d_fullres) on fold 0 of splits/patient_splits.json, one GPU, time-budgeted.
#   sh scripts/nnunet_run.sh check|prepare|train|test
# env: GPU (default 1), TRAINER (default nnUNetTrainerBudget; nnUNetTrainerBudget_smoke = 4-epoch check),
#      NNUNET_BUDGET_MIN (epoch minutes, default 210), HARD_STOP (train wall limit, default 240m)
set -eu
cd "$(dirname "$0")/.."
export nnUNet_raw="$PWD/nnunet/raw" nnUNet_preprocessed="$PWD/nnunet/preprocessed" nnUNet_results="$PWD/nnunet/results"
export nnUNet_compile=f MPLCONFIGDIR="$PWD/.cache/mpl"   # torch.compile: triton cache fails on the NFS home (Errno 61)
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-20}" CUDA_VISIBLE_DEVICES="${GPU:-1}"
BIN=.venv/bin D=801 DS=Dataset801_Pancreas P=nnUNetResEncUNetMPlans C=3d_fullres TR="${TRAINER:-nnUNetTrainerBudget}"
R="$nnUNet_results/$DS/${TR}__${P}__$C" OUT="results/nnunet/$TR"

case "$1" in
  check)    # T1 alignment on all 80 DICOM cases; nnunet_prepare.py refuses to run unless the rule is identity
    mkdir -p nnunet/dicom                       # flat view: nnunet/dicom/PANCREAS_XXXX -> its single DICOM series folder
    for p in data/pancreas_ct/PANCREAS_*; do
      ln -sfn "$PWD/$(dirname "$(find "$p" -name '*.dcm' | head -1)")" "nnunet/dicom/$(basename "$p")"
    done
    $BIN/python tools/check_alignment.py --dicom nnunet/dicom --raw_labels data/lables --out results/alignment_all80 ;;
  prepare)
    $BIN/python tools/nnunet_prepare.py
    $BIN/nnUNetv2_plan_and_preprocess -d $D -pl nnUNetPlannerResEncM -c $C -np 16 --verify_dataset_integrity ;;
  train)    # --c resumes from checkpoint_latest.pth (starts fresh if none); ends with full-volume validation
    ln -sf "$PWD/src/nnunet_trainer.py" "$($BIN/python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/"
    timeout "${HARD_STOP:-240m}" $BIN/nnUNetv2_train $D $C 0 -tr "$TR" -p $P --c ;;
  test)     # post-processing chosen on VALIDATION only, then applied to the 16 test cases
    $BIN/nnUNetv2_determine_postprocessing -i "$R/fold_0/validation" -ref "$nnUNet_preprocessed/$DS/gt_segmentations" \
      -plans_json "$R/plans.json" -dataset_json "$R/dataset.json"
    $BIN/nnUNetv2_predict -i "$nnUNet_raw/$DS/imagesTs" -o "$OUT/test_raw" -d $D -c $C -f 0 -tr "$TR" -p $P \
      -chk checkpoint_final.pth
    $BIN/nnUNetv2_apply_postprocessing -i "$OUT/test_raw" -o "$OUT/test_pp" -pp_pkl_file "$R/fold_0/validation/postprocessing.pkl" \
      -plans_json "$R/plans.json" -dataset_json "$R/dataset.json"
    cp "$R/fold_0/validation/summary.json" "$OUT/val_summary_nnunet.json"
    cp "$R/fold_0/training_log_"*.txt "$R/fold_0/progress.png" "$OUT/"
    $BIN/python tools/nnunet_eval.py --pred "$OUT/test_pp" --gt "$nnUNet_raw/$DS/labelsTs" --out "$OUT" --name test ;;
  *) echo "usage: $0 check|prepare|train|test" >&2; exit 2 ;;
esac
