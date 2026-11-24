# E0 — Early-Trimester Standard nnUNetv2 2D Baseline

Version: `e0-baseline-v1.0`
Status: complete (5/5 folds, 300 epochs each)

## What this is

A standard nnUNetv2 2D baseline trained from scratch on early-trimester fetal cardiac
ultrasound, used to answer: *how well can a plain nnUNetv2 model support early-trimester
cardiac-axis measurement?*

The only change from stock nnUNet is `num_epochs = 300` instead of 1000.
Multi-center and prospective cohorts were held out and never used for training,
model selection, or post-processing tuning.

## Contents

| File | Content |
|---|---|
| `README.md` | this file |
| `report.md` | full experiment report: setup, 5-fold training, internal/external metrics, Ne/Po split, comparison with the historical reference, failures, conclusion |
| `config.json` | exact configuration actually used: nnUNetv2 version, trainer, 2D plan, epochs, folds, TTA, inference, split version |
| `metrics/segmentation_metrics.csv` | Dice/IoU per label and HD95, per fold and per evaluation set |
| `metrics/angle_metrics.csv` | Full Angle MAE/MedAE/P90/P95, Card and Reference axis error, per set, cohort and Ne/Po |

## Headline numbers

| Set | N | Dice | Full Angle MAE | MedAE | P95 |
|---|---|---|---|---|---|
| Internal 5-fold | 621 | 0.9354 | 3.64 | 2.67 | 9.92 |
| Multicenter | 442 | 0.9267 | 4.09 | 3.30 | 10.51 |
| Prospective | 411 | 0.9325 | 3.36 | 2.89 | 8.01 |
| External total | 853 | 0.9295 | 3.74 | 3.08 | 9.35 |

Ne (normal) is usable: internal MAE 3.29. Po (pathological) is not yet: internal MAE 7.62.

## Not in this repository

Deliberately excluded (kept on the training server only):

- original ultrasound images, gold masks, historical inference masks
- TAR archives and manual measurement JSON
- case-level result tables (they contain identifiers)
- predicted masks and model weights (3.5 GB)

Weights are at
`local/nnunet_results/Dataset501_EarlyRetroCardiac/nnUNetTrainer_300Epochs__nnUNetPlans__2d/fold_{0..4}/checkpoint_{best,final}.pth`
with SHA256 checksums listed in `report.md` section 9.

## Reproduce

See `report.md` section 11. Code lives in `scripts/` and `src/`.
A GPU and the private dataset are required; this archive alone is not runnable.
