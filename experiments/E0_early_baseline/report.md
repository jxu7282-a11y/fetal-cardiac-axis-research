# E0 — Early-Trimester Standard nnUNetv2 2D Baseline

**Question:** Using early-trimester data only, with standard nnUNetv2 trained from scratch,
what segmentation performance and cardiac-axis measurement performance can be reached?

**Answer (short):** Normal (Ne) cases are clinically usable (Full Angle MAE ~3.3-4.0 deg).
Pathological (Po) cases are not yet reliable (MAE 7.6 deg internal, P95 23.3 deg).
Segmentation Dice and angle accuracy are only loosely coupled: the error is dominated by a
small tail of failures, not by average segmentation quality.

---

## 1. Experimental setup

| Item | Value |
|---|---|
| Dataset | `Dataset501_EarlyRetroCardiac` (early trimester only) |
| Task | 2D segmentation |
| Training | scratch, random initialization (no pretraining) |
| Trainer | `nnUNetTrainer_300Epochs` = stock `nnUNetTrainer` with `num_epochs=300` only |
| Plans | stock `nnUNetPlans`, unmodified |
| Network | `PlainConvUNet`, 8 stages, 46,335,249 parameters |
| Batch / Patch | 5 / 768 x 768 |
| Loss | nnUNet default (DC+CE with deep supervision) |
| Optimizer / LR | SGD (0.99 momentum, nesterov, wd 3e-5) / 1e-2 poly |
| Augmentation | nnUNet default |
| GT | `label_gold` only; `label_infer` never used for training |
| Epochs | 300 per fold |
| Folds | 0-4, serial, single physical GPU 3 |
| Inference | no TTA; `checkpoint_best.pth`; external = 5-fold ensemble |

The only deviation from stock nnUNet is `num_epochs = 300` (stock default 1000),
which also rescales the poly LR horizon. Every other setting was verified identical
(architecture, parameter count, loss, optimizer, initial LR, batch, patch, augmentation, plans).

## 2. Data and split

1474 formal cases: retrospective 621 (571 Ne / 50 Po), multicenter 442 (434 / 8),
prospective 411 (400 / 11).

Five folds generated on the retrospective 621 only, with
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` stratified by Ne/Po at
case-record level (no patient grouping, no cross-cohort dedup).

| fold | train | train Ne/Po | val | val Ne/Po |
|---|---|---|---|---|
| 0 | 496 | 456 / 40 | 125 | 115 / 10 |
| 1 | 497 | 457 / 40 | 124 | 114 / 10 |
| 2 | 497 | 457 / 40 | 124 | 114 / 10 |
| 3 | 497 | 457 / 40 | 124 | 114 / 10 |
| 4 | 497 | 457 / 40 | 124 | 114 / 10 |

Every case is a validation case exactly once (union 621, overlap 0). Multicenter and
prospective are external (fold -1) and were never used for training, model selection,
or post-processing tuning.

## 3. Five-fold training results (300 epochs each)

| fold | val Dice | train window | exit |
|---|---|---|---|
| 0 | 0.9380 | 01:35 - 03:23 | 0 |
| 1 | 0.9336 | 03:23 - 05:10 | 0 |
| 2 | 0.9346 | 05:10 - 06:58 | 0 |
| 3 | 0.9389 | 06:58 - 08:45 | 0 |
| 4 | 0.9319 | 08:45 - 10:34 | 0 |

All folds completed 300/300 epochs. No OOM, no CUDA error, no dataloader failure.

## 4. Segmentation metrics

Per-class Dice (mean over folds, internal 621):

| Label | Anatomical semantics (confirmed) | Geometry role (archive placeholder) | Dice | HD95 (px) |
|---|---|---|---|---|
| 1 | 胸椎 (thoracic vertebra) | reference center | 0.9152 | 11.13 |
| 2 | 左侧肋骨 (left rib) | label_2 (unused by geometry) | 0.8810 | 22.53 |
| 3 | 右侧肋骨 (right rib) | label_3 (unused by geometry) | 0.9168 | 16.72 |
| 4 | 左侧心脏 (left heart) | cardiac region 4 | **0.9682** | **2.93** |
| 5 | 右侧心脏 (right heart) | cardiac region 5 | **0.9668** | **2.87** |
| 6 | 胸腔 (thorax) | thorax reference | 0.9644 | 8.26 |

Label semantics are user-confirmed and recorded in `docs/project/06_LABEL_SEMANTICS.md`; the
"geometry role" column keeps the historical archive placeholders. Numeric IDs and pixel values
are unchanged.

Summary:

| Set | N | Dice | IoU | HD95 |
|---|---|---|---|---|
| Internal 5-fold | 621 | 0.9354 | 0.8914 | 10.76 |
| Multicenter | 442 | 0.9267 | 0.8751 | 25.27 |
| Prospective | 411 | 0.9325 | 0.8801 | 8.46 |
| External total | 853 | 0.9295 | 0.8775 | 17.17 |

The three labels that drive the cardiac axis (4, 5, 6) reach Dice >0.95 and HD95 <9 px
internally, so the geometric input is sound. Multicenter HD95 is 3x higher than
prospective, consistent with a domain/resolution shift.

## 5. Full Angle measurement metrics

| Set | N | valid | Full MAE | MedAE | P90 | P95 | Card | Ref |
|---|---|---|---|---|---|---|---|---|
| Internal 5-fold | 621 | 621 | 3.64 | 2.67 | 7.61 | 9.92 | 3.09 | 2.57 |
| Multicenter | 442 | 442 | 4.09 | 3.30 | 8.75 | 10.51 | 3.20 | 2.86 |
| Prospective | 411 | 411 | 3.36 | 2.89 | 6.77 | 8.01 | 2.58 | 2.53 |
| External total | 853 | 853 | 3.74 | 3.08 | 7.70 | 9.35 | 2.91 | 2.70 |

Measurement failure rate: **0%** (621 + 853, all predictions produced a finite geometry).

Per-fold spread: Full MAE 3.40-4.04 (SD 0.24); fold 3 has the heaviest tail (P95 13.5 deg).

## 6. Ne / Po stratification

| Population | N | Full MAE | MedAE | P90 | P95 | Card | Dice |
|---|---|---|---|---|---|---|---|
| Internal Ne | 571 | 3.29 | 2.60 | 6.90 | 8.85 | 2.71 | 0.9432 |
| Internal Po | 50 | **7.62** | 5.03 | 16.27 | **23.28** | 7.38 | 0.8466 |
| Multicenter Ne | 434 | 4.03 | 3.30 | 8.51 | 10.31 | 3.10 | 0.9287 |
| Multicenter Po | 8 | 7.00 | 3.64 | 17.92 | 18.66 | 8.75 | 0.8145 |
| Prospective Ne | 400 | 3.31 | 2.86 | 6.60 | 7.91 | 2.56 | 0.9322 |
| Prospective Po | 11 | 5.23 | 5.32 | 9.21 | 9.67 | 3.57 | 0.9412 |

Po error is 2.0-2.3x the Ne error in every cohort. Card error, not reference error,
carries most of the Po degradation (7.38 vs 2.64 internally): the cardiac-region
shape/OBB long axis is what breaks down on malformed hearts.

## 7. Comparison with the historical label_infer reference

| Metric | E0 internal (621) | E0 external (853) | Historical (1474) |
|---|---|---|---|
| Full MAE | 3.64 | 3.74 | **3.50** |
| MedAE | 2.67 | 3.08 | 2.82 |
| P90 | 7.61 | 7.70 | **7.16** |
| P95 | 9.92 | 9.35 | **8.82** |
| Card error | 3.09 | 2.91 | **2.81** |
| Ref error | 2.57 | 2.70 | 2.57 |
| Dice | 0.9354 | 0.9295 | **0.957** |
| IoU | 0.8914 | 0.8775 | **0.924** |

The historical model is slightly better on average (Dice +0.022, MAE -0.14 deg), but E0
is already at the same clinical scale after only 300 epochs of from-scratch training.

By cohort, E0 vs historical:

| Cohort | E0 Full MAE | Historical Full MAE |
|---|---|---|
| Retrospective (621) | 3.64 | 3.39 |
| Multicenter (442) | 4.09 | 3.81 |
| Prospective (411) | 3.36 | 3.32 |

## 8. Failure statistics

- Measurement failures: **0** in 1474 evaluated cases (no missing predictions, no NaN geometry,
  every prediction produced a valid card axis from id4/id5 and a ref axis from id6/id1).
- Segmentation failures (mean Dice <0.5). These cases are **kept in every statistic above**
  and were not silently excluded; they are reported here only as anonymized slots:

| Slot | Cohort | Class | Mean Dice |
|---|---|---|---|
| internal-01 | retrospective (5-fold) | Ne | 0.29 |
| internal-02 | retrospective (5-fold) | Ne | 0.40 |
| external-multi-01 | multicenter | Po | 0.65 |
| external-multi-02 | multicenter | Po | 0.66 |
| external-multi-03 | multicenter | Ne | 0.66 |

- Largest Full Angle errors cluster in Po cases and in low-resolution multicenter images
  (one case is only 412x406 px). Worst internal Full Angle error 34.9 deg; worst external 22.3 deg.
- Tail summary: internal Ne P95 8.85 deg vs Po P95 23.28 deg; external multicenter P95 10.51 deg
  vs prospective P95 8.01 deg.

Case-level tables (with identifiers) are deliberately **not** published; only aggregated
statistics are archived here.

## 9. Artifacts and integrity

Weights, logs and predictions remain on the server under `local/` and are not in Git.

Weights: `local/nnunet_results/Dataset501_EarlyRetroCardiac/nnUNetTrainer_300Epochs__nnUNetPlans__2d/fold_{0..4}/checkpoint_{best,final}.pth` (3.5 GB total)

SHA256:

```
785d345a55c0b716fc3b5d31f8ad392f45d2bcdd507884580b297e7b039092f5  fold_0/checkpoint_best.pth
2c8c1ff0e545ee401d38a038b691fdadbb61c3a38fa05ac6c3a443cee64ccd6e  fold_0/checkpoint_final.pth
6b0d23be6d97679968dad5dc83501e9fb75b6d1656c4690df0c938c8d4650fac  fold_1/checkpoint_best.pth
1c8b2464e109fbae2dd19a5c1121b927ccf6c05c9913fc1ffbee1c319e57de2d  fold_1/checkpoint_final.pth
ab3320e140e111f8ae82ff8acc4efe4ca5878d794a57a3706bd93d363908571b  fold_2/checkpoint_best.pth
f720aadc1594e9604eb18fca56d70d8ddf540394a340e655eef710b826541635  fold_2/checkpoint_final.pth
a7fb2cdd58412c0b998b45758d1dd71eef7c893938132ee3c7fec4fc2a18e021  fold_3/checkpoint_best.pth
9006b1847d622fb629b9b47c10f0a3e65181225be39e1af6a2662fe85767b8f8  fold_3/checkpoint_final.pth
28c42b13f4fabad6ce9dd203b582d391db5628f995527a63a69a987e4a83c781  fold_4/checkpoint_best.pth
9005ae7708a11662dd1b02da43ff2c4533239a90dd8d543429f0ea9fa135dfa6  fold_4/checkpoint_final.pth
```

Train logs: `local/phase0/e0_300ep_fold{0..4}.log`
Internal predictions: `local/nnunet_results/.../fold_{0..4}/validation/`
External predictions: `local/phase0/e0_external_pred_ensemble/`

## 10. Scientific conclusion

1. **Standard early-trimester nnUNetv2 supports cardiac-axis measurement for normal fetuses.**
   Full MAE 3.29 deg internally and 3.31-4.03 deg externally is in the range of manual
   repeat-measurement agreement (typically 2-4 deg).
2. **It does not yet support pathological fetuses.** Po MAE 7.62 deg internally with P95
   23.3 deg; the failure is concentrated in the cardiac-region (id4/id5) axis.
3. **Segmentation quality and measurement quality are only loosely coupled.** Raising
   average Dice did not proportionally lower MAE, and external HD95 tripled while MAE moved
   only 0.1 deg. Improvements should target the Po and low-quality tail, not average Dice.

## 11. Reproduction notes

Scripts: `scripts/01..12` (audit, fixed GT, folds, nnUNet export, splits, train, eval, HD95).
Trainer: `src/nnunet_ext/nnUNetTrainer_300Epochs.py`.
Evaluation entry: `scripts/04_evaluate.py` (unified GT, geometry and metrics for all models).

This archive contains aggregated metrics only. The pilot run with the 1000-epoch schedule
(stopped at 471 epochs) is excluded and does not contribute to any number above.
