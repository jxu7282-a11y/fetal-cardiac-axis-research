# M1 — Measurement-aware Segmentation with Dual-axis Auxiliary Supervision

**Version:** `m1-dual-axis-v1.0`
**Archived:** 2026-11-26
**Status:** training complete (5/5 folds × 300 epochs), evaluation and statistics complete

---

## 1. Hypothesis and position in the project

The final clinical objective is early-trimester cardiac-axis **angle** measurement; segmentation
is only a means. E0 established a from-scratch nnUNetv2 2D baseline; E2/E3 tested mid-trimester
*transfer* (whole-network / encoder-only), both returning neutral-to-negative clinical results.

M1 is a **different, non-transfer** idea, run in parallel to E3:

> **H-M1 (measurement-aware supervision):** attaching two lightweight regression heads
> (Card axis, Reference axis) to the final decoder feature and training them jointly with the
> segmentation loss injects *measurement-relevant* inductive bias into the shared representation,
> which then lowers Full Angle error after the frozen geometry pipeline — even though inference
> uses only the segmentation logits.

M1 trains from scratch on `Dataset501` (no mid-trimester data, no pretrained weights). It is a
method ablation against the E0 scratch baseline, with E2/E3 as secondary references.

**Recorded answer: the hypothesis is not supported.** Internal Full Angle is significantly worse
than E0 and E3; external is null. The auxiliary heads collapse to a near-constant predictor
(§8). This is an honest negative result.

## 2. Data

| Cohort | N | Ne / Po | Role |
|---|---:|---|---|
| Retrospective | 621 | 571 / 50 | five-fold development, OOF evaluation |
| Multicenter | 442 | 434 / 8 | external test |
| Prospective | 411 | 400 / 11 | prospective test |
| **Total** | **1474** | 1405 / 69 | |

- Splits are the frozen E0 five-fold (`splits_final.json`, not regenerated); `fold` column
  matches E0 fold assignment 100% for all 621 internal cases.
- Ground truth is the frozen E0 `label_gold` + TAR manual lines; `label_infer` never used.
- External/prospective cohorts were never used for training, epoch selection, or tuning.
- **Scratch init:** no mid-trimester images, masks, or weights are read anywhere in M1.

## 3. Network architecture

M1 wraps the **unchanged E0 `PlainConvUNet`** (3-channel input, 7-channel segmentation output,
deep supervision) and adds two auxiliary heads on the **final decoder stage feature** — the same
tensor that feeds `seg_layers[-1]`.

```
PlainConvUNet(x) -> seg logits (7 ch, deep supervision)        # E0, unchanged
                  |
   final decoder feature [B, 32, H, W]
        |-> Card head:     AdaptiveAvgPool2d(1) -> Flatten -> Linear(32, 2) -> [B,2]
        |-> Reference head:AdaptiveAvgPool2d(1) -> Flatten -> Linear(32, 2) -> [B,2]
```

- Two heads are **independent** (separate Linear layers, 66 params each, 132 total).
- Output is an **un-normalised double-angle vector** `[cos 2θ, sin 2θ]`; it is L2-normalised
  inside the loss.
- **Segmentation head initialisation is identical to E0**; auxiliary heads use their own random
  init, so they do not perturb the E0 parameter initialisation.
- **Inference contract:** `network(x)` returns **only** the original segmentation logits. The
  auxiliary heads are ignored at inference, post-processing and geometry. `forward_with_aux(x)`
  exists only for the training step.

## 4. Loss

Double-angle encoding, angles in degrees converted to radians:

```
q(θ) = [cos 2θ, sin 2θ]                       (θ and θ+180° map to the same q — undirected axis)
L_axis = 1 - dot(q_pred_normalised, q_gt)      (cosine loss, range [0, 2])
L_aux  = (L_card + L_ref) / 2                   (each averaged over its own valid samples)
L_total = L_seg + 0.1 * L_aux
```

- `L_seg` is the **E0 inherited** CompoundLoss (Dice + CE) with deep supervision — unchanged.
- Missing / NaN axis labels only mask the **corresponding** auxiliary loss; the case is **not**
  dropped from segmentation training. A batch with no valid aux labels contributes exactly zero
  aux loss (guarded against NaN by a `pred.sum()*0` zero path).

## 5. Training configuration

| Item | Value |
|---|---|
| nnUNetv2 | 2.8.1 (torch 2.7.1+cu118) |
| Trainer | `nnUNetTrainer_M1DualAxis` (parent `nnUNetTrainer_300Epochs`, E0) |
| Configuration | 2d, `nnUNetPlans` (E0 originals) |
| Epochs | 300, folds 0–4, scratch from epoch 0 |
| Optimizer / LR | inherited E0 default (SGD momentum 0.99 nesterov; 0.01 PolyLR) |
| Batch / patch | 5 / [768, 768] (E0) |
| Augmentation | inherited E0 2D (rotation, mirroring, scaling, gamma; **no elastic**) — orientation labels transformed with the **actual** SpatialTransform/MirrorTransform parameters (§7) |
| Physical GPU | **1** (GPU 3 reserved for E3) |
| Results root | `local/m1_measurement_aware/results_nnunet/Dataset501_EarlyRetroCardiac/nnUNetTrainer_M1DualAxis__nnUNetPlans__2d` |

**Five-fold completeness (all 5/5 folds reached 300 epochs, exit 0):**

| Fold | best epoch | final epoch | checkpoint_best | checkpoint_final |
|---|---:|---:|---|---|
| 0 | 288 | 300 | present | present |
| 1 | 258 | 300 | present | present |
| 2 | 289 | 300 | present | present |
| 3 | 300 | 300 | present | present |
| 4 | 290 | 300 | present | present |

All 10 checkpoints (`checkpoint_best.pth`, `checkpoint_final.pth` per fold) load cleanly and
contain the 4 `aux_heads.*` tensors. Weights stay on the server (not archived).

## 6. Evaluation protocol

Strictly reuses the E0/E3 pipeline:

- **GT** = frozen E0 fixed clinical manual GT (`local/phase0/fixed_gt_v1.csv`); never modified.
- **Geometry** = frozen E0: Card axis from ID4/ID5 OBB weighted long axis; Reference axis from
  ID6→ID1 centroid; Full Angle = acute angle between them. Unchanged.
- **Internal**: five-fold OOF, each fold's `checkpoint_best.pth` predicts its own validation split.
- **External**: 5-fold ensemble of `checkpoint_best.pth` over imagesTs (853), `--disable_tta`.
- **Metrics**: Full Angle MAE/MedAE/P90/P95, ≥10°/≥15° rate, Card/Reference MAE, Dice/IoU/HD95,
  measurement failure rate. Stratified by Ne/Po and multicenter/prospective.
- **Paired bootstrap**: 10000 resamples, seed 42, 95% percentile CI, paired by case uid (local only).

## 7. Augmentation ↔ orientation-label consistency

The E0 2D pipeline uses rotation, mirroring and scaling (**no elastic**). Because the aux targets
are directions, they must rotate/reflect/scale with the image. `M1DataLoader` calls each
transform's `get_parameters` **once** and applies the **same** parameters to image, mask and the
axis vector, then re-encodes the double angle:

- Rotation / anisotropic scaling: the axis **direction vector** is pushed through the **actual
  forward affine matrix** of `SpatialTransform`, then re-encoded as `q(θ)`.
- Horizontal/vertical mirroring: direction reflected about the real mirror axis.
- Intensity transforms: leave direction labels unchanged.

The forward (input→output) matrix is distinguished from the inverse sampling matrix; this was
verified against nnUNet 2.8.1 / batchgenerators source (not guessed). CPU unit tests cover
θ≡θ+180°, 0/180° boundary, rotation, h/v mirror, anisotropic scaling, and a synthetic line pushed
through a real `SpatialTransform`. No untransformed-angle path is used.

## 8. Results

### 8.1 Internal OOF (621 cases; 620 valid Full Angle)

| Model | Full MAE | MedAE | P90 | P95 | ≥10° % | Card MAE | Ref MAE | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **M1** | **3.815** | 2.806 | 8.132 | **10.084** | 5.16 | 3.135 | 2.530 | 0.9329 | 0.8873 |
| E0 | 3.639 | 2.665 | 7.606 | 9.923 | 4.99 | 3.088 | 2.575 | 0.9354 | 0.8914 |
| E2 | 3.607 | 2.662 | 7.457 | 9.460 | 4.19 | 3.112 | 2.458 | 0.9337 | 0.8878 |
| E3 | 3.647 | 2.806 | 7.176 | 9.173 | 4.35 | 3.188 | 2.404 | 0.9339 | 0.8882 |

### 8.2 External (853 cases; 5-fold ensemble, TTA off)

| Model | Full MAE | MedAE | P90 | P95 | ≥10° % | Card MAE | Ref MAE | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **M1** | 3.692 | 3.039 | 7.707 | 9.369 | 3.52 | 2.830 | 2.747 | 0.9289 | 0.8769 |
| E0 | 3.739 | 3.077 | 7.700 | 9.351 | 3.87 | 2.906 | 2.701 | 0.9295 | 0.8775 |
| E2 | 3.860 | 3.048 | 7.664 | 9.506 | 4.34 | 2.704 | 3.085 | 0.9275 | 0.8735 |
| E3 | 3.693 | 3.004 | 7.342 | 8.848 | 2.81 | 2.768 | 2.922 | 0.9282 | 0.8746 |

By cohort (M1): **Multicenter 442 — Full MAE 4.012°, P95 10.167°, Dice 0.9258. Prospective
411 — Full MAE 3.347°, P95 8.184°, Dice 0.9322.**

### 8.3 Axis decomposition (M1 vs E0/E3)

Axis MAE is essentially flat; the Full Angle deterioration is not a simple single-axis failure.

- Internal Card MAE: M1 3.135 vs E0 3.088 vs E3 3.188 → M1 slightly **better** than E3, no
  significant change vs E0 (paired +0.051°, CI [−0.067, +0.169]).
- Internal Ref MAE: M1 2.530 vs E0 2.575 vs E3 **2.404** → M1 significantly **worse than E3**
  (paired +0.126°, CI [0.005, 0.299]); vs E0 slightly better (−0.049°, ns).
- Po subgroup (n=50): Card flat (M1 7.316 vs E3 7.343); **Ref worse** (M1 2.789 vs E3 2.456,
  paired +0.333° [0.005, 0.689]); Full Δ vs E3 = +1.743° [0.516, 3.174].

Because Full Angle = |Card − Ref| is a difference functional, the net Full Angle harm exceeds any
single-axis change (single-axis substitution counterfactuals explain only ~20% of the internal
delta; the rest is a joint two-axis drift, consistent with correlated signed axis errors of +0.59).

### 8.4 Paired bootstrap (Δ = M1 − baseline, negative = M1 better; 10000 resamples, seed 42)

| Scope | Subset | Comparison | n | Δ mean | 95% CI | Significant |
|---|---|---|---:|---:|---|---|
| Internal | ALL | M1 vs **E0** | 620 | **+0.181°** | **[0.023, 0.359]** | **yes (worse)** |
| Internal | ALL | M1 vs E2 | 620 | +0.208° | [0.06, 0.37] | yes (worse) |
| Internal | ALL | M1 vs **E3** | 620 | **+0.168°** | **[0.009, 0.337]** | **yes (worse)** |
| Internal | Ne | M1 vs E0 | 570 | +0.152° | [−0.00, 0.32] | no |
| Internal | **Po** | **M1 vs E3** | 50 | **+1.743°** | **[0.516, 3.174]** | **yes (worse)** |
| Internal | Po | M1 vs E0 | 50 | +0.511° | [−0.38, 1.48] | no |
| External | ALL | M1 vs E0 | 853 | −0.048° | [−0.151, 0.057] | no |
| External | ALL | M1 vs E3 | 853 | −0.001° | [−0.204, 0.179] | no |
| External | multicenter | M1 vs E0 | 442 | −0.076° | [−0.220, 0.072] | no |
| External | prospective | M1 vs E0 | 411 | −0.018° | [−0.151, 0.140] | no |

Full detail in `metrics/paired_bootstrap.csv`.

### 8.5 Error burden and per-case movement

- ≥10° errors: internal M1 32 (5.16%) vs E0 31 (4.99%) vs E3 27 (4.35%); external M1 30
  (3.52%) vs E0 33 (3.87%) vs E3 24 (2.81%).
- Per-case ±1° bands (internal): M1 vs E0 111 improved / 126 worsened / 383 stable; M1 vs E3
  122 / 146 / 352. Po (vs E3): 10 improved / 27 worsened / 13 stable.
- **Statistical fairness:** M1 and E3 share the **same** single failing case (ID1 dropout →
  Reference undefined), so M1−E0 and M1−E3 both use the **same 620** valid cases; E0 alone has
  621. Recomputing E0 on the identical 620 (excluding that one case, E0 Full error 6.60°) leaves
  the M1−E0 delta unchanged at +0.181° — denominators are not the source of the effect.

### 8.6 Cohort matching audit

All four models share identical uid sets (621 internal, 853 external), zero duplicate uids, zero
invalid predictions, and bit-identical GT. The only NaN is the single internal measurement
failure, reported (not dropped) below. No case silently disappeared.

## 9. Diagnostic: why the auxiliary supervision failed

The negative result is **explained**, not mysterious. Three converging pieces of evidence
indicate the auxiliary heads collapsed to a near-constant predictor.

### 9.1 Auxiliary loss sits just below the "constant predictor" baseline

`L_axis = 1 − cos(q_pred, q_gt) ∈ [0, 2]`. A predictor that **ignores the image** and always
emits the training-set mean direction achieves `L = 1 − |E[e^{i2θ}]|`. Computed per fold on each
fold's actual training set:

| Fold | const-baseline mean | observed aux loss (last epoch) | excess over baseline |
|---|---:|---:|---:|
| 0 | 0.669 | 0.727 | +0.058 |
| 1 | 0.684 | 0.745 | +0.061 |
| 2 | 0.672 | 0.731 | +0.059 |
| 3 | 0.672 | 0.743 | +0.071 |
| 4 | 0.673 | 0.736 | +0.063 |

After 300 epochs the aux loss is only **~0.06 below a no-image constant predictor**. `L = 0.73`
corresponds to a double-angle error of ~74°, i.e. an axis error of ~37° (chance ≈ 45°). The
heads learned almost nothing about direction.

### 9.2 The heads' default output equals the label mean

The normalised direction of each `aux_heads.*.fc.bias` is the network's default axis angle.
Across all ten heads it is **87–96°**, versus label marginal means of Card 90.0° / Ref 87.3°:

| Fold | Card bias dir | Ref bias dir |
|---|---:|---:|
| 0 | 89° | 87° |
| 1 | 88° | 96° |
| 2 | 94° | 87° |
| 3 | 92° | 89° |
| 4 | 93° | 87° |

The weights (norm 1.0–1.2 on a 32-d linear layer) never developed directional selectivity.

### 9.3 Mechanism: GAP deletes the spatial information an axis needs

`AdaptiveAvgPool2d(1)` collapses `[B, 32, H, W]` → `[B, 32]`, discarding the spatial layout that
**defines** an axis direction. A global mean feature cannot encode "where ID4→ID5 points". The two
independent heads therefore both regress to the dataset mean direction. The 0.1× aux gradient
still flows into the shared decoder but now acts as a direction-agnostic (≈90°-prior)
regulariser, not useful axis supervision — consistent with the observed correlated two-axis drift
and the small segmentation perturbations (§10).

This is an **architectural failure**, not an optimisation or schedule problem: more epochs or a
larger loss weight cannot recover information the pooling already threw away.

## 10. Dice vs angle

- Dice is at ceiling (~0.93). M1 vs E0 Dice Δ = −0.0025 (p=8.5e−8, statistically real but
  clinically negligible); vs E3 Δ = −0.0010 (p=0.039).
- Per-case ΔDice vs ΔFull Angle: Pearson −0.184 (p=4e−6) but Spearman −0.005 (p=0.90) — the
  linear correlation is driven entirely by a few outliers; **there is no monotone Dice–angle
  coupling**. 60.3% of cases have |ΔDice| < 0.005.
- Cases with Dice down but angle better: 11; Dice up but angle worse: 8 (internal, n=620).
  Angle change is **not** driven by segmentation accuracy — reinforcing that Dice must never be
  used as the M1 success criterion.

## 11. The one measurement failure

One internal case (fold 0, Ne) has `valid_pred=True`, Card axis computed, **Reference axis NaN**
because the predicted mask contains zero ID1 pixels (Reference axis = ID6→ID1 centroid). The
**same case also fails under E3**. It is retained and reported, not dropped:
`metrics/failure_cases_anonymized.csv` (`case_hash` only; real uid stays in `local/`). It affects
only that case's Ref/Full values; the other 620 internal cases are unaffected. Measurement failure
rate: internal 1/621 (0.16%), external 0.

## 12. Scientific conclusion

1. **No demonstrated clinical gain, and internal harm.** Internal Full Angle is significantly
   worse than both E0 (+0.181°, CI [0.023, 0.359]) and E3 (+0.168°, CI [0.009, 0.337]); external
   is statistically null versus E0 (−0.048°) and E3 (−0.001°).
2. **Internal vs external inconsistency.** The harm is internal-only; externally M1 is, if
   anything, marginally better than E0. This "OOF-sensitive / ensemble-smoothed" pattern means the
   negative effect does not reproduce in the deployed-style external cohorts — i.e. its clinical
   weight is small either way.
3. **Axis attribution.** Card axis does not degrade; the Reference axis is significantly worse
   than E3 (Po +0.333°). Full Angle harm is amplified by the |Card−Ref| difference geometry and a
   correlated two-axis drift, not by a single failed head.
4. **Pathological (Po) subgroup.** A real but moderate negative shift vs E3 (+1.743°, n=50,
   direction-consistent across folds) that is **not** significant vs E0 and **reverses** on the
   external Po cases. Read as exploratory, not a systematic "pathology breakage".
5. **Root cause is architectural.** GAP → both aux heads collapse to a ~constant mean-direction
   predictor (§9). The measurement-aware *idea* was never actually given directional information.
6. **Verdict:** the current M1 design (GAP + Linear heads) should **not** be kept as-is. The
   scaffolding (double-angle targets, augmentation-synced labels, loss, trainer, zero-inference-
   cost integration) is sound and reusable; the **pooling** is the failing component.

## 13. Limitations

- Single seed, single architecture, single dataset; no run-to-run variance estimate. Fold 0 alone
  accounts for ~53% of the internal M1−E0 delta sum, so part of the internal effect is fold-level
  noise at this effect size (<0.2°).
- Po is n=50 internal / 19 external — subgroup effects are exploratory; the M1−E3 Po effect loses
  most of its magnitude (−1.74° → −1.29°) when the single worst case is removed.
- Auxiliary loss is logged only as a **batch mean**; per-epoch Card/Ref split and valid counts are
  not in the logs, so per-case correlation of aux error with measurement error was not possible
  without modifying the trainer (out of scope).
- Internal (OOF) and external (ensemble) protocols differ; cross-cohort differences may reflect
  protocol as well as true generalisation.
- This experiment reports a null/negative result; it does not explore spatial-preserving
  pooling or other fixes (that is future work, not part of M1).

## 14. Where the artifacts live

Archived in Git (this directory): report, README, config, aggregate metrics CSVs and summary
JSONs, anonymised failure record, aggregate cohort audit, and the M1 source + CPU unit tests.

Server-only, never uploaded: model weights (`checkpoint_best/final.pth` × 5 folds), prediction
masks, raw case-level CSVs with real uids, training logs, and any per-case manifests. Paths:

- training: `local/m1_measurement_aware/results_nnunet/`
- evaluation: `local/m1_measurement_aware/evaluation/`
- analysis: `local/m1_measurement_aware/analysis/` (incl. `M1_analysis_report.md`, `M1_diagnostic_report.md`)
