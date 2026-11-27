#!/usr/bin/env bash
# HET-PSM: mid-trimester pretraining (side head) -> early encoder-only transfer.
# NOT executed in step 2. Folds run serially on one physical GPU.
set -euo pipefail
cd /data5/xjx/zaoyun/data/project1

export nnUNet_raw=$PWD/local/nnunet_raw
export nnUNet_preprocessed=$PWD/local/nnunet_preprocessed
export nnUNet_results=$PWD/local/het_psm/results_nnunet
export nnUNet_extTrainer=$PWD/experiments
export MPLCONFIGDIR=$PWD/local/mplcache
export CUDA_VISIBLE_DEVICES=3

PHASE=${1:?usage: run_het_psm.sh mid|early}
OUT=$PWD/local/het_psm
mkdir -p "$OUT/logs" "$OUT/encoder_audit" "$nnUNet_results"

if [ "$PHASE" = mid ]; then
  TRAINER=nnUNetTrainer_HETPSMMid
  DATASET=503
  CONFIG=2d
  LOG=$OUT/logs/hetpsm_mid_fold_all.log
  nnUNetv2_train "$DATASET" "$CONFIG" all -tr "$TRAINER" 2>&1 | tee "$LOG"
elif [ "$PHASE" = early ]; then
  export HETPSM_MID_CHECKPOINT=${HETPSM_MID_CHECKPOINT:?set to the HET-PSM mid checkpoint}
  export HETPSM_AUDIT_DIR=$OUT/encoder_audit
  TRAINER=nnUNetTrainer_HETPSMEarly
  DATASET=501
  CONFIG=2d
  for FOLD in 0 1 2 3 4; do
    LOG=$OUT/logs/hetpsm_early_fold${FOLD}.log
    CT=""
    FOLDDIR=$nnUNet_results/Dataset501_EarlyRetroCardiac/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
    if [ -f "$FOLDDIR/checkpoint_latest.pth" ] || [ -f "$FOLDDIR/checkpoint_final.pth" ]; then
      CT="-c"
    fi
    nnUNetv2_train "$DATASET" "$CONFIG" "$FOLD" -tr "$TRAINER" $CT 2>&1 | tee "$LOG"
  done
else
  echo "unknown phase: $PHASE" >&2
  exit 2
fi
