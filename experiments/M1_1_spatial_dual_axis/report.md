# M1.1 — Spatial-preserving Dual-axis Auxiliary Supervision

**Version:** `m1.1-spatial-dual-axis-v1.0`
**Archived:** 2026-11-28
**Status:** training complete (5/5 folds × 300 epochs), formal evaluation and auxiliary-head analysis complete

---

## 1. Position in the project

The clinical objective is early-trimester cardiac-axis **angle** measurement; segmentation is
only a means. E0 established a from-scratch nnUNetv2 2D baseline; E2/E3 tested mid-trimester
transfer (neutral-to-negative). M1 tested measurement-aware auxiliary supervision with global
average pooling and produced an honest negative result: its two Card/Reference heads collapsed to
a near-constant predictor because `AdaptiveAvgPool2d(1)` destroys the spatial layout an axis needs.

**M1.1 is a minimal controlled follow-up** that changes *only* that pooling. It asks:

> *If the auxiliary heads can actually see spatial layout (4×4 pooling), do they learn a real,
> generalisable Card/Reference axis predictor — and does that, in turn, lower the Mask-derived
> Full Angle error?*

**Recorded answer:** yes to the first (the aux heads now learn genuine axes), no to the second (no
significant Full Angle gain over E0/E2/E3/M1; the mask-derived geometry remains the better axis
estimator).

## 2. Data

| Cohort | N | Ne / Po | Role |
|---|---:|---|---|
| Retrospective | 621 | 571 / 50 | five-fold development, OOF evaluation |
| Multicenter | 442 | 434 / 8 | external test |
| Prospective | 411 | 400 / 11 | prospective test |
| **Total** | **1474** | 1405 / 69 | |

- Splits are the frozen E0 five-fold (`splits_final.json`, not regenerated).
- Ground truth is the frozen `fixed_gt_v1.csv` (`label_gold` + TAR manual lines); `label_infer`
  never used.
- External/prospective cohorts were never used for training, selection, or tuning.
- Scratch init: no mid-trimester images, masks, or weights are read anywhere in M1.1.

## 3. Experimental design: M1 → M1.1

The only architectural change is the auxiliary-head pooling and its FC input width:

```
M1    Card/Ref head: AdaptiveAvgPool2d(1)        -> Flatten -> Linear(32, 2)     (66 params/head)
M1.1  Card/Ref head: AdaptiveAvgPool2d((4, 4))   -> Flatten -> Linear(512, 2)    (1026 params/head)
```

Everything else is inherited unchanged: the E0 `PlainConvUNet` backbone (bit-identical build,
396 backbone tensors verified), 7-channel deep-supervised segmentation output, the dual-axis
Card/Reference task with the same double-angle labels, the loss and its 0.1 weight, the data
pipeline and augmentation with synced axis-label transforms, folds, 300 epochs, and
initialisation. The inference contract is unchanged: `network(x)` returns **only** segmentation
logits; the auxiliary heads are used only during training.

**Verification suite:**
- **21 CPU unit tests** cover the spatial heads and the trainer
  (`test_spatial_aux_heads.py`, `test_m1_1_trainer.py`), all OK.
- **Smoke test** (GPU 3, one shortened epoch) passes: backbone and segmentation output identical
  to M1, `pool_size = 4`, `aux_fc_in = 512`, aux params 2052, output `[B,2]` per head, finite
  loss/gradients, non-zero aux gradients, non-constant aux output (unit std max 0.468), aux
  params updated.
- **Backbone-consistency check** confirms the main network parameter set is identical to a
  same-seed M1/E0 build, so any difference is attributable to the pooling change alone.

## 4. Training and formal evaluation

**Five-fold completeness — all 5/5 folds reached 300 epochs, exit 0:**

| Fold | start → end | exit | mean val Dice | best saved epoch | final epoch |
|---|---|---|---:|---:|---:|
| 0 | 2026-11-27 01:50 → 07:09 | 0 | 0.9377 | 300 | 300 |
| 1 | 07:09 → 10:33 | 0 | 0.9323 | 298 | 300 |
| 2 | 10:33 → 16:09 | 0 | 0.9364 | 283 | 300 |
| 3 | 16:09 → 21:13 | 0 | 0.9392 | 297 | 300 |
| 4 | 21:13 → 2026-11-28 00:23 | 0 | 0.9325 | 298 | 300 |

- `checkpoint_best.pth` (EMA-best) and `checkpoint_final.pth` exist and load for every fold; both
  expose 400 network tensors (14 seg-layer + 4 aux-head entries) matching E0 backbone + dual heads.
- Checkpoints are read-only here; SHA256 per fold is in `config.json` (`training.checkpoint_sha256`)
  and §12.

**Evaluation protocol (identical to E0/E2/E3/M1):**
- GT `fixed_gt_v1.csv`; folds `fold_v2.csv`; geometry unchanged (`src/cardiac_axis/evaluate.py`,
  fixed_geometry); HD95 = scipy surface-distance 95th percentile over labels 1..6.
- **Internal:** each fold's `checkpoint_best` predicts its own validation split → OOF union of 621
  retrospective cases. Each case predicted only by a model that never trained on it.
- **External:** 5-fold `checkpoint_best` ensemble with `--disable_tta` → multicenter 442 +
  prospective 411 = 853 cases.
- TTA disabled to keep internal/external protocols identical.
- UID sets verified identical to E0 for every model/scope; 0 duplicate UIDs; GT manual axes
  identical across models; 0 measurement failures, 0 NaN for M1.1.
- Predictions were written to an isolated directory; no other experiment, GT, fold, or checkpoint
  was modified.

## 5. Core results

### 5.1 Full Angle (degrees) and segmentation

| Scope | Model | N | Full MAE | MedAE | P90 | P95 | ≥10° % | Card MAE | Ref MAE | Dice | HD95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Internal | M1.1 | 621 | 3.700 | 2.737 | 7.389 | 10.02 | 5.15 | 3.155 | 2.603 | 0.9329 | 11.09 |
| Internal | E0 | 621 | 3.639 | 2.665 | 7.606 | 9.92 | 4.99 | 3.088 | 2.575 | 0.9354 | 11.09 |
| Internal | E2 | 621 | 3.607 | 2.662 | 7.457 | 9.46 | 4.19 | 3.112 | 2.458 | 0.9337 | 11.09 |
| Internal | E3 | 621 | 3.647 | 2.806 | 7.176 | 9.17 | 4.35 | 3.188 | 2.404 | 0.9339 | 11.09 |
| Internal | M1 | 621 | 3.815 | 2.807 | 8.132 | 10.08 | 5.16 | 3.135 | 2.530 | 0.9329 | 11.09 |
| External | M1.1 | 853 | 3.719 | 2.976 | 7.444 | 9.65 | 4.22 | 2.948 | 2.911 | 0.9295 | 16.51 |
| External | E0 | 853 | 3.739 | 3.077 | 7.700 | 9.35 | 3.87 | 2.906 | 2.701 | 0.9295 | 16.51 |
| External | E2 | 853 | 3.860 | 3.048 | 7.664 | 9.51 | 4.34 | 2.704 | 3.085 | 0.9275 | 16.51 |
| External | E3 | 853 | 3.693 | 3.004 | 7.342 | 8.85 | 2.81 | 2.768 | 2.922 | 0.9282 | 16.51 |
| External | M1 | 853 | 3.692 | 3.039 | 7.707 | 9.37 | 3.52 | 2.830 | 2.747 | 0.9289 | 16.51 |

HD95 is dominated by the multicenter cohort (~24 mm; a cohort property shared by all models;
prospective ~8.4 mm). Segmentation is at ceiling and on par with E0/E3/M1, not better.

### 5.2 Paired bootstrap vs E0/E2/E3/M1 (10000 resamples, seed 42)

Δ = M1.1 − baseline (negative = M1.1 better). **No Full Angle comparison is statistically
significant; every 95% CI includes 0.**

| Scope | vs | Δ Full MAE | 95% CI | p |
|---|---|---:|---|---:|
| Internal | E0 | +0.061 | [−0.106, +0.248] | 0.50 |
| Internal | E2 | +0.093 | [−0.085, +0.280] | 0.33 |
| Internal | E3 | +0.054 | [−0.131, +0.244] | 0.59 |
| Internal | M1 | −0.114 | [−0.291, +0.071] | 0.22 |
| External | E0 | −0.021 | [−0.164, +0.153] | 0.76 |
| External | E2 | −0.141 | [−0.360, +0.061] | 0.18 |
| External | E3 | +0.026 | [−0.171, +0.199] | 0.75 |
| External | M1 | +0.027 | [−0.119, +0.206] | 0.78 |

M1.1 partially reverses M1's internal regression (M1 3.815 → M1.1 3.700, Δ −0.114) and lowers
M1's internal Po P95 (26.8 → 26.7), but neither reaches significance. (Note: a few single-axis
comparisons do reach p<0.05 — e.g. internal Ref vs E2/E3 — but they are not consistent across
scopes and are not the primary endpoint; see `metrics/paired_bootstrap.csv`.) The
improvement/worsening split is ~50/50 for every comparison (e.g. internal vs E0: 52.5% improved,
47.5% worsened), i.e. no systematic per-case advantage.

### 5.3 Stratification (Ne/Po, Multicenter/Prospective)

Full Angle MAE / P95 by stratum (M1.1):

| Scope | Stratum | N | Full MAE | Full P95 |
|---|---|---:|---:|---:|
| Internal | Ne | 571 | 3.377 | 8.61 |
| Internal | Po | 50 | 7.389 | 26.71 |
| External | Multicenter | 442 | 4.106 | 10.33 |
| External | Prospective | 411 | 3.303 | 7.94 |

All Ne/Po and multicenter/prospective paired comparisons are non-significant versus the baselines
(`metrics/stratified_metrics.csv`, `metrics/paired_bootstrap.csv`).

### 5.4 Failures and valid counts

- M1.1: **0 measurement failures** across 621 internal and 853 external cases; 0 NaN
  card/ref/full errors; 0 invalid predictions or shapes.
- E3 and M1 each had 1 internal full-angle-unavailable case; M1.1 loses none.
- No failure was silently dropped; `metrics/error_threshold_counts.csv` records all
  valid denominators and ≥10°/≥15°/failure counts.

## 6. Auxiliary-head independent analysis

This section uses the fold checkpoints **read-only** (no training). Auxiliary outputs are
decoded with `angle = atan2(v1, v0)/2 (mod 180)`; internal = per-fold OOF (621), external = 5-fold
ensemble in double-angle space (853), TTA off. Aggregation is in the axial double-angle space,
never an arithmetic mean of angles. All 621/853 cases extracted with 0 failures.

### 6.1 Auxiliary head vs manual GT

| Scope | Axis | N | MAE | MedAE | P90 | P95 | Bias | r | R² | CCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Internal | Card | 621 | 4.40 | 3.48 | 8.68 | 11.37 | +0.20 | 0.969 | 0.938 | 0.969 |
| Internal | Ref | 621 | 4.23 | 3.27 | 8.88 | 10.15 | −0.07 | 0.807 | 0.612 | 0.806 |
| External | Card | 853 | 4.25 | 3.34 | 8.52 | 10.91 | +0.70 | 0.989 | 0.978 | 0.989 |
| External | Ref | 853 | 4.22 | 3.23 | 8.76 | 11.09 | +0.02 | 0.849 | 0.701 | 0.849 |

- Card axis very strong (r≈0.97–0.99), Reference strong but clearly weaker (r≈0.81–0.85).
- **Not constant:** predicted std matches GT std (internal card pred 36.4 vs GT 36.6; ref 44.1 vs
  44.0). Best constant predictor gives axis MAE ≈31–41°; the aux head achieves ≈4.2–4.4° —
  ~7–9× better. The 4×4 pooling fixed M1's GAP collapse.
- Internal (OOF) and external (ensemble) accuracy are nearly identical (Full MAE 5.60 vs 5.73),
  so this is not an internal-only artefact.

### 6.2 Auxiliary head vs Mask-derived measurement (paired)

Δ = aux − mask (negative would favour aux). **Mask-derived is significantly better on every
axis/cohort (all CIs exclude 0):**

| Scope | Metric | Aux MAE | Mask MAE | Δ | 95% CI | aux better / mask better |
|---|---|---:|---:|---:|---|---|
| Internal | Card | 4.40 | 3.16 | +1.24 | [+0.91, +1.59] | 240 / 381 |
| Internal | Ref | 4.23 | 2.60 | +1.63 | [+1.21, +2.02] | 200 / 421 |
| Internal | Full | 5.60 | 3.70 | +1.90 | [+1.50, +2.30] | 201 / 420 |
| External | Card | 4.25 | 2.95 | +1.30 | [+0.97, +1.62] | 312 / 541 |
| External | Ref | 4.22 | 2.91 | +1.31 | [+0.97, +1.66] | 322 / 531 |
| External | Full | 5.73 | 3.72 | +2.01 | [+1.56, +2.47] | 336 / 517 |

The mask decoder is directly optimised to produce the id4/id5/id6/id1 structures the fixed geometry
consumes, so it reads the axis from a sharp, task-aligned segmentation; the aux head reads a coarse
4×4-pooled feature through a weak 0.1-weighted objective.

### 6.3 Auxiliary loss and constant floor

Per-fold auxiliary loss drops from ~0.75 at start to ~0.005 in the last 10 epochs (all folds),
whereas M1 stalls at ~0.73–0.75 throughout. Constant-prediction floors (best-constant
`1 − dot(q_const, q_gt)`) are ~0.62–0.91 depending on axis/cohort. M1.1's ~0.005 is far below any
constant floor, confirming real learning; M1's ~0.74 sits *at* the floor. (Per project rule, aux
loss alone is not treated as evidence of clinical success.)

### 6.4 Exploratory Po finding

On the hardest Po subgroup the aux head has a **lower worst-case tail** than the mask geometry in
several strata even though its mean is worse: e.g. internal Po aux Full P95 15.1° vs mask 26.7°;
external multicenter Po aux card MAE 11.8 vs mask 14.4 (mask P95 34.6°). This is exploratory
(small N: internal Po 50, external Po 19) and the best-constant floors there are high, but it
motivates a Po-tail fusion direction rather than global fusion.

### 6.5 Conclusions of the auxiliary analysis

- The 4×4 spatial pooling **did fix the M1 auxiliary-head collapse**: the heads now encode genuine
  per-case Card (strong) and Reference (solid) axes, non-constant, far above the constant floor.
- This genuine auxiliary learning **does not translate into a Full Angle measurement benefit**
  because (i) the aux head is a *weaker* axis estimator than the mask geometry (Δ≈+1.3°/axis,
  +1.9–2.0° Full), (ii) it is attached to a shared decoder at weight 0.1 and cannot redirect the
  segmentation path, and (iii) the clinical Full Angle is computed from the **mask**, not the aux
  head.
- **Mask geometry is overall the better estimator; the aux head does not beat it.** The only place
  the aux head shows an advantage is the Po lower tail (exploratory).

## 7. Scientific conclusion

1. **Statistical fact:** M1.1 is statistically indistinguishable from E0, E2, E3 and M1 on Full
   Angle MAE/P95 and Card/Reference single-axis error, in internal OOF and external cohorts (paired
   bootstrap 10000, seed 42; all CIs cross zero). Segmentation is on par with E0/E3/M1, not better.
2. **Mechanistic inference:** the 4×4 spatial pooling fixed the M1 auxiliary-head collapse (aux
   loss now far below the constant floor, non-constant predictions), confirming the stated
   mechanism — but the auxiliary objective is not wired tightly enough to the segmentation/geometry
   path to move the clinical endpoint.
3. **Did M1.1 meet expectations?** As a *mechanism* check, yes (aux collapse fixed). As a *clinical
   measurement* improvement, no: no significant Full Angle gain over E0/E3/M1, and it does not beat
   the E0 baseline. Negative/neutral for the measurement objective, with a useful positive
   mechanistic finding.

## 8. Limitations

- Single seed, single architecture, single dataset; no run-to-run variance estimate. Effects are
  all <0.2° Full Angle and fold-level noise can dominate at this size.
- Po is small (internal 50, external 19); the Po-tail observation is exploratory.
- Internal (OOF) and external (ensemble) protocols differ; cross-cohort differences may reflect
  protocol as well as true generalisation.
- Auxiliary loss is logged only as a batch mean; per-case correlation of aux error with
  measurement error is not from the training logs (the independent aux-head analysis in §6 is a
  separate read-only extraction).
- M1.1 reports a neutral/negative measurement result; it does not explore aux↔mask fusion (future
  work).

## 9. Privacy

Archived in Git (this directory): report, README, config, aggregate metrics CSVs and summary JSONs,
aggregate cohort audit, and the M1.1 source + CPU unit tests. All case-level files, raw auxiliary
extractions (`aux_raw_*.csv`), `aux_head_case_level.csv`, checkpoints, prediction masks, logs, and
any uid / case_hash / patient names remain server-only under `local/m1_1_spatial_dual_axis/`. No
absolute server path or private directory mapping is included beyond the relative `local/...`
locations needed for reproduction.

## 10. Where the artifacts live (server-only)

- training: `local/m1_1_spatial_dual_axis/results_nnunet/`
- evaluation: `local/m1_1_spatial_dual_axis/evaluation/`
- analysis: `local/m1_1_spatial_dual_axis/analysis/`
- logs: `local/m1_1_spatial_dual_axis/logs/`

## 11. Reproduce

Inference uses `checkpoint_best.pth`, TTA off; internal per-fold OOF, external 5-fold ensemble in
double-angle space. See `local/m1_1_spatial_dual_axis/evaluation/` scripts and
`local/m1_1_spatial_dual_axis/analysis/` scripts. A GPU and the private dataset are required; the
archive alone is not runnable.

## 12. Checkpoint SHA256 (best/final, per fold)

```
c35db1219903e6ebea7cb618aadd5024731edf218cdf064805c114add11b0d76  fold_0/checkpoint_best.pth
e3e1073c5273d68ca9229d93271da883b913ed3bb8ec0aba5556c79bca8e8eaf  fold_1/checkpoint_best.pth
5afbfb219dcaf3e74038011ae69c7d17945d2c0822bd7c2ba899e36388d2bf9e  fold_2/checkpoint_best.pth
c524f304b2f5eda794f88955248c1907f634796098526844ce80ebed59d70039  fold_3/checkpoint_best.pth
a52dc4cf4a131873749c7415a3239b1f552f067b8904d1fc2aada9af18a7aafd  fold_4/checkpoint_best.pth
e671a53a9ebfa0dededae39e89750b985c6d60d056cffa626e5e20db57aaba89  fold_0/checkpoint_final.pth
31ab81956550cb7c3bb6849cd9196904b219798fefd71dfed612d47fabb92fbe  fold_1/checkpoint_final.pth
ff4093e15ed18c3ba3fbba2cff6d64cd1fead7b175ba95b5b6e905fc2314ec01  fold_2/checkpoint_final.pth
3549cf6e821ff4f73bd1fabb807db83f3a62c65592cd3a6db03d22ed87b15a60  fold_3/checkpoint_final.pth
21bb2d8a21c5b097b29c7a4ef647d8383da0a94d642a436645209a2d0737b8b4  fold_4/checkpoint_final.pth
```
