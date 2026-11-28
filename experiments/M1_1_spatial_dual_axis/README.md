# M1.1 — Spatial-preserving Dual-axis Auxiliary Supervision

Version: `m1.1-spatial-dual-axis-v1.0`
Status: complete (5/5 folds × 300 epochs; formal evaluation and auxiliary-head analysis done)

## What this is

M1 attached two Card/Reference axis-regression heads to the final decoder feature with
`AdaptiveAvgPool2d(1)`. Its diagnostic showed those heads collapsed to a near-constant
predictor (aux loss stuck at the no-image constant floor, direction pinned near 90°): global
average pooling deletes the spatial layout that an axis direction requires.

M1.1 is the minimal controlled follow-up that changes **only** the pooling:

```
M1    Card/Ref head: AdaptiveAvgPool2d(1)        -> Flatten -> Linear(32, 2)
M1.1  Card/Ref head: AdaptiveAvgPool2d((4, 4))   -> Flatten -> Linear(512, 2)
```

Everything else is inherited from M1: backbone, segmentation heads, data pipeline, loss and
weight, folds, epochs, initialisation and inference contract. Not a transfer experiment;
trained from scratch on `Dataset501`.

The hypothesis: 4×4 spatial pooling lets the auxiliary heads learn a non-constant axis while
every other experimental condition stays fixed. The clinical question remains whether this
translates into a lower **Mask-derived Full Angle** error.

**Recorded answer:** the auxiliary-head collapse **is fixed** (aux heads now predict real
per-case axes), but there is **no demonstrated clinical measurement improvement** — M1.1 is
statistically indistinguishable from E0/E2/E3/M1 on Full Angle, and the mask-derived geometry
remains a better axis estimator than the auxiliary heads themselves.

## Contents

| File | Content |
|---|---|
| `README.md` | this file |
| `report.md` | design, training, formal evaluation, paired statistics, stratified results, auxiliary-head independent analysis, conclusion, limitations |
| `config.json` | exact configuration actually used, five-fold completeness, checkpoint SHA256, headline results, verdict |
| `metrics/segmentation_metrics.csv` | Dice / IoU / HD95 per label by scope and subgroup |
| `metrics/angle_metrics.csv` | Full Angle MAE/MedAE/P90/P95, Card and Ref axis error, per set/cohort/Ne-Po |
| `metrics/formal_comparison_table.csv` | M1.1/E0/E2/E3/M1 × scope × subgroup × metric |
| `metrics/paired_bootstrap.csv` | paired bootstrap Δ vs E0/E2/E3/M1, 10000 resamples, seed 42, 95% CI |
| `metrics/stratified_metrics.csv` | Ne/Po and multicenter/prospective strata |
| `metrics/improvement_worsening_counts.csv` | per-case improved/worsened ≥1° bands |
| `metrics/error_threshold_counts.csv` | ≥10°, ≥15° and measurement-failure counts |
| `metrics/internal_summary.json` | internal OOF aggregate (621) with Ne/Po and HD95-by-label |
| `metrics/external_summary.json` | external aggregate (853) plus per-cohort |
| `metrics/cohort_match_audit.json` | M1.1/E0/E2/E3/M1 uid-set, duplicate, GT, NaN audit (aggregate only) |
| `metrics/aux_loss_curve_summary.csv` | per-fold auxiliary loss trajectory (M1.1 vs M1) |
| `metrics/const_baseline_axis_loss.csv` | best-constant double-angle loss floor per axis/cohort |
| `metrics/aux_head_metrics.csv` | aux-head Card/Ref accuracy vs manual GT (MAE/MedAE/P90/P95/Bias/r/R²/CCC) |
| `metrics/aux_head_stratified.csv` | aux-head vs mask-derived axis/Full MAE and P95 by stratum |
| `metrics/aux_head_const_baseline.csv` | constant-prediction baseline angle MAE per axis/cohort |
| `metrics/aux_vs_mask_comparison.csv` | paired aux-head vs mask-derived comparison with 95% CI |
| `dual_axis_aux_heads.py`, `m1_1_trainer.py` | `SpatialAxisHead` / `SpatialDualAxisAuxHeads`, `nnUNetTrainer_M1_1SpatialDualAxis` |
| `tests/` | 21 CPU unit tests (`test_spatial_aux_heads.py`, `test_m1_1_trainer.py`) |

## Headline numbers

| | Full MAE | Full P95 | Card MAE | Ref MAE | Dice |
|---|---:|---:|---:|---:|---:|
| Internal M1.1 (n=621) | 3.700 | 10.02 | 3.155 | 2.603 | 0.9329 |
| Internal E0 | 3.639 | 9.92 | 3.088 | 2.575 | 0.9354 |
| Internal M1 | 3.815 | 10.08 | 3.135 | 2.530 | 0.9329 |
| External M1.1 (n=853) | 3.719 | 9.65 | 2.948 | 2.911 | 0.9295 |
| External E0 | 3.739 | 9.35 | 2.906 | 2.701 | 0.9295 |
| External M1 | 3.692 | 9.37 | 2.830 | 2.747 | 0.9289 |

Paired bootstrap (Δ = M1.1 − baseline, negative = M1.1 better; all CIs include 0):

- Internal vs E0 +0.061 [−0.106, 0.248]; vs E3 +0.054 [−0.131, 0.244]; vs M1 −0.114 [−0.291, 0.071].
- External vs E0 −0.021 [−0.164, 0.153]; vs E3 +0.026 [−0.171, 0.199].

**Bottom line:** no significant Full Angle gain anywhere; M1.1 partially reverses M1's internal
regression but not significantly. Auxiliary heads (independent of the mask) reach Card r≈0.97–0.99
/ Ref r≈0.81–0.85 (MAE≈4.2–4.4°), far above any constant baseline — so the GAP collapse is fixed —
but mask-derived geometry is still significantly better (aux Full MAE 5.6–5.7° vs mask 3.7°).

## Not in this repository

Deliberately excluded (kept on the training server only, under `local/m1_1_spatial_dual_axis/`):

- original ultrasound images, gold masks, historical inference masks
- TAR archives and manual measurement JSON
- case-level result tables (they contain identifiers), `aux_raw_*.csv`, `aux_head_case_level.csv`
- predicted masks, model weights (checkpoint_best/final × 5 folds), training logs
- any absolute server path or private directory mapping

Weights live at
`local/m1_1_spatial_dual_axis/results_nnunet/Dataset501_EarlyRetroCardiac/nnUNetTrainer_M1_1SpatialDualAxis__nnUNetPlans__2d/fold_{0..4}/checkpoint_{best,final}.pth`
with SHA256 checksums listed in `config.json` (`training.checkpoint_sha256`) and `report.md`.

## Reproduce

See `report.md` §11. Inference is `checkpoint_best.pth`, TTA off, internal per-fold OOF and
external 5-fold ensemble. A GPU and the private dataset are required; this archive alone is not
runnable.
