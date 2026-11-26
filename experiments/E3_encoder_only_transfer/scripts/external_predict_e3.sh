#!/usr/bin/env bash
# E3 external prediction: 853 cases (multicenter 442 + prospective 411),
# E3 5-fold ensemble of checkpoint_best.pth, TTA disabled.
# Mirrors E0 scripts/11_predict_external_e0.sh and E2's external script exactly
# except nnUNet_results -> E3 results, output -> E3 evaluation dir, -tr -> E3 trainer.
set -u
cd "$(dirname "$0")/../../.." || exit 1

export nnUNet_raw=$PWD/local/nnunet_raw
export nnUNet_preprocessed=$PWD/local/nnunet_preprocessed
export nnUNet_results=$PWD/local/e3_encoder_only_transfer/results_nnunet
export MPLCONFIGDIR=$PWD/local/mplcache
export nnUNet_extTrainer=$PWD/src/nnunet_ext
export CUDA_VISIBLE_DEVICES=${E3_GPU:-3}

OUTDIR=$PWD/local/e3_encoder_only_transfer/evaluation/external_pred_ensemble
LOG=$PWD/local/e3_encoder_only_transfer/evaluation/logs/external_predict.log
mkdir -p "$OUTDIR"
mkdir -p "$(dirname "$LOG")"

echo "START $(date +%F' '%T) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset501_EarlyRetroCardiac/imagesTs" \
  -o "$OUTDIR" \
  -d 501 -c 2d -f 0 1 2 3 4 \
  -tr nnUNetTrainer_E3EncoderOnly -chk checkpoint_best.pth --disable_tta > "$LOG" 2>&1
RC=$?
echo "EXIT_CODE=$RC END $(date +%F' '%T)"
echo "predicted files: $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)"
exit $RC
