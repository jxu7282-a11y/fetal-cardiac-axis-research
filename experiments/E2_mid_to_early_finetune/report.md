# E2 — Mid-Trimester Pretraining → Early-Trimester Fine-Tuning: Report

**Question:** Does initializing an early-trimester nnUNetv2 segmentation model from a
mid-trimester pretrained checkpoint improve early-trimester cardiac-axis measurement
(Full Angle error) versus the E0 from-scratch baseline?

**Answer (neutral / null result):** No clinically meaningful improvement. Segmentation is
essentially unchanged (Dice Δ ≈ −0.001 to −0.002; HD95 Δ ≤ 0.5 px). Full Angle error is
statistically indistinguishable from E0 in every cohort; the only nominally favourable
movement is on the small, high-variance pathological (Po) subset and on the P95 tail, neither
reaching robust significance. **Full Angle MAE and P95 — not Dice — are the primary criteria.**
All deltas below are Δ = E2 − E0 (negative = E2 better).

This archive reports **prediction + evaluation only**; E0 artifacts, folds, GT, and
post-processing were never modified. All numbers are taken verbatim from the existing frozen
reports and aggregate CSVs; nothing was recomputed or re-interpreted for this archive.

---

## 1. Setup and integrity

| Item | Value |
|---|---|
| Target dataset | `Dataset501_EarlyRetroCardiac`, 2D, 3-channel, labels 0–6, 621 retrospective train |
| nnU-Net | 2.8.1, configuration `2d`, trainer `nnUNetTrainer_300Epochs`, plans `nnUNetPlans` |
| E2 folds | 0–4 (E0 original split, not regenerated); each fold independently initialized from the same mid checkpoint; all 300 epochs |
| Mid checkpoint SHA256 | `df94a1d9aac1557a50d63856d684ef236508b45731dbcfd3479c85493314cd9f` |
| Internal predictions | each fold's own validation set → OOF over 621 |
| External predictions | 5-fold ensemble of `checkpoint_best.pth` over external images, `--disable_tta` |
| GT / geometry | frozen E0: fixed gold-assisted manual GT; `ID4/ID5` OBB weighted card axis, `ID6→ID1` centroid reference axis |
| Cohorts | Multicenter 442 (434 Ne / 8 Po), Prospective 411 (400 Ne / 11 Po) |
| Measurement failures | 0 / 1474 |

The evaluation pipeline is the same as E0 (same GT, folds, geometry, angle math, case
inclusion). Reproduction check: applying it to E0's own predictions recovered E0's published
internal Full MAE 3.6386 / P95 9.9235 and external 3.7395 / 9.3509 exactly.

## 2. Weight-transfer contract (382 / 382 + 14 reinit)

Loader: official `nnunetv2.run.load_pretrained_weights.load_pretrained_weights` via
`nnUNetv2_train ... -pretrained_weights`. Status **PASS** for all five folds.

| Contract item | Expected | Actual |
|---|---:|---:|
| Total network keys | 396 | 396 |
| Non-segmentation keys | 382 | 382 |
| Non-segmentation keys transferred (value-equal) | 382 | **382** |
| Missing / extra non-segmentation keys | 0 | **0** |
| Non-segmentation shape mismatches | 0 | **0** |
| Segmentation keys skipped / reinitialized | 14 | **14** |

The 14 `decoder.seg_layers.{0..6}.{weight,bias}` were skipped because the pretrained head has
10 output classes vs E0's 7, and were reinitialized. **Optimizer / epoch state was NOT
restored** — fine-tuning starts fresh training state from pretrained network weights. Each
fold's training log begins with the same load header from the single mid checkpoint (no
chaining between folds). Full detail: `pretrained_loading_audit.md` (server-side).

## 3. Segmentation metrics (E2 vs E0)

| Set | N | Dice E2 / E0 | IoU E2 / E0 | HD95 E2 / E0 |
|---|---:|---:|---:|---:|
| Internal 5-fold OOF | 621 | 0.9337 / 0.9354 | 0.8878 / 0.8914 | 10.74 / 10.76 |
| Multicenter | 442 | 0.9247 / 0.9267 | 0.8710 / 0.8751 | 24.25 / 25.27 |
| Prospective | 411 | 0.9304 / 0.9325 | 0.8762 / 0.8801 | 8.60 / 8.46 |
| External total | 853 | 0.9275 / 0.9295 | 0.8735 / 0.8775 | 16.71 / 17.17 |

Per-label Dice (internal, E2): id1 0.9146, id2 0.8804, id3 0.9152, id4 0.9658, id5 0.9638,
id6 0.9626. Geometry-driving labels (4/5/6) remain >0.96. E2 is marginally *lower* on Dice/IoU
(≈ −0.002 to −0.004) with mixed HD95 — all within noise. (Per-label/per-fold detail:
`metrics/segmentation_metrics.csv`.)

## 4. Full Angle measurement metrics (E2 vs E0)

Primary clinical endpoint; Δ = E2 − E0.

| Set | Pop | N | E2 MAE | E0 MAE | Δ MAE [95% CI] | E2 P95 | E0 P95 | Δ P95 [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Internal 5-fold | ALL | 621 | 3.607 | 3.639 | −0.031 [−0.170, 0.111] | 9.460 | 9.924 | −0.464 [−1.372, 0.398] |
| Internal 5-fold | Ne | 571 | 3.349 | 3.290 | +0.059 [−0.063, 0.182] | 8.518 | 8.849 | −0.331 [−0.884, 0.479] |
| Internal 5-fold | Po | 50 | 6.557 | 7.619 | **−1.062 [−2.162, −0.187]** | 21.804 | 23.278 | −1.475 [−15.04, 4.12] |
| External total | ALL | 853 | 3.860 | 3.739 | +0.120 [−0.082, 0.356] | 9.506 | 9.351 | +0.155 [−0.646, 0.911] |
| External total | Ne | 834 | 3.822 | 3.689 | +0.133 [−0.068, 0.370] | 9.324 | 9.191 | +0.133 [−0.651, 0.876] |
| External total | Po | 19 | 5.526 | 5.976 | −0.449 [−1.704, 0.739] | 13.555 | 17.496 | −3.940 [−6.442, 1.732] |
| Multicenter | ALL | 442 | 4.317 | 4.088 | +0.229 [−0.152, 0.676] | 10.617 | 10.507 | +0.111 [−1.078, 1.492] |
| Multicenter | Ne | 434 | 4.284 | 4.034 | +0.250 [−0.135, 0.687] | 10.380 | 10.309 | +0.071 [−0.953, 1.330] |
| Multicenter | Po | 8 | 6.071 | 7.000 | −0.929 [−3.321, 0.971] | 16.815 | 18.662 | −1.848 [−6.442, 1.605] |
| Prospective | ALL | 411 | 3.368 | 3.365 | +0.004 [−0.123, 0.126] | 7.915 | 8.013 | −0.098 [−0.495, 0.605] |
| Prospective | Ne | 400 | 3.320 | 3.313 | +0.006 [−0.116, 0.129] | 7.854 | 7.908 | −0.054 [−0.398, 0.730] |
| Prospective | Po | 11 | 5.130 | 5.231 | −0.101 [−1.392, 1.444] | 9.257 | 9.673 | −0.416 [−2.919, 3.047] |

MedAE (E2 / E0): internal 2.662 / 2.665; multicenter 3.202 / 3.304; prospective 2.906 / 2.888;
external 3.048 / 3.077. **Reading:** normal (Ne) fetuses show no benefit (internal +0.059,
external +0.133); Po shows the only favourable MAE shift (internal −1.06, CI excludes 0) but n
is tiny (50 / 8 / 19) with wide CIs; P95 moves favorably in most internal strata but not
significantly. (`metrics/angle_metrics.csv`.)

## 5. Paired bootstrap (5000×, seed 42, 95% percentile CI)

| Population | n paired | Δ MAE | 95% CI | Δ P95 | 95% CI |
|---|---:|---:|---|---:|---|
| Internal total | 621 | −0.031 | [−0.170, +0.111] | −0.464 | [−1.372, +0.398] |
| External total | 853 | +0.120 | [−0.081, +0.360] | +0.155 | [−0.638, +0.940] |
| Internal Ne | 571 | +0.059 | [−0.063, +0.182] | −0.331 | [−0.884, +0.479] |
| Internal Po | 50 | −1.062 | [−2.162, −0.187] | −1.475 | [−15.04, +4.12] |
| External Ne | 834 | +0.133 | [−0.068, +0.370] | +0.133 | [−0.651, +0.876] |
| External Po | 19 | −0.449 | [−1.704, +0.739] | −3.940 | [−6.442, +1.732] |

Both total MAE and both total P95 CIs include zero → **no population-level difference**. The
only CI excluding zero is internal Po MAE (n=50, wide P95 CI) — suggestive, not decision-grade.
(`metrics/E2_vs_E0_comparison.csv`.)

## 6. Case-level distribution (paired)

Threshold |ΔFullError| < 1° = "stable".

| Population | N | improved ≥1° | worsened ≥1° | stable <1° | improved (sign) | worsened (sign) |
|---|---:|---:|---:|---:|---:|---:|
| Internal | 621 | 121 (19.5%) | 108 (17.4%) | 392 (63.1%) | 308 (49.6%) | 313 (50.4%) |
| External | 853 | 171 (20.0%) | 157 (18.4%) | 525 (61.5%) | 435 (51.0%) | 418 (49.0%) |
| All | 1474 | 292 (19.8%) | 265 (18.0%) | 917 (62.2%) | 743 (50.4%) | 731 (49.6%) |

The population mean is flat because improvements and degradations roughly cancel — a
**redistribution** of error rather than a uniform gain.

## 7. Geometry decomposition: Card axis vs Reference axis

This is the key mechanistic result. Mean axial error vs fixed manual GT; Δ = E2 − E0.

| Population | n | ΔCard | ΔRef | ΔFull |
|---|---:|---:|---:|---:|
| All | 1474 | **−0.107** | **+0.173** | +0.056 |
| Internal | 621 | +0.024 | −0.117 | −0.031 |
| External | 853 | −0.202 | +0.384 | +0.120 |
| Internal Ne | 571 | +0.087 | −0.109 | +0.059 |
| Internal Po | 50 | −0.697 | −0.217 | −1.062 |
| External Ne | 834 | −0.197 | +0.352 | +0.133 |
| External Po | 19 | −0.429 | +1.800 | −0.449 |
| Multicenter Ne | 434 | −0.366 | +0.706 | +0.250 |
| Hard E0≥10° (64) | 64 | −1.082 | −0.517 | −1.993 |
| Severe negative transfer (5) | 5 | −1.117 | +36.533 | +36.186 |
| External excl. severe-5 | 848 | −0.197 | +0.171 | −0.092 |

**E2 improves the Cardiac axis but harms the Reference axis.** The severe negative-transfer
cases show cardiac −1.12° (better) but reference +36.5° (catastrophic). (`metrics/axis_error_comparison.csv`.)

## 8. Gold Mask vs Infer Mask (geometric floor vs segmentation error)

| Population | Gold Card | E0 Card | E2 Card | Gold Ref | E0 Ref | E2 Ref | Gold Full | E0 Full | E2 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 2.99 | 2.98 | 2.88 | 2.66 | 2.65 | 2.82 | 3.83 | 3.70 | 3.75 |
| External | 3.10 | 2.91 | 2.70 | 2.78 | 2.70 | 3.09 | 4.10 | 3.74 | 3.86 |
| Hard E0≥10° | 7.13 | 11.04 | 9.96 | 3.02 | 6.47 | 5.95 | 9.41 | 14.18 | 12.19 |

- **Reference axis has a geometric floor:** even the *Gold Mask* gives Ref error 2.5–3.1°
  (two far-apart centroids → fragile to ID6/ID1 shape). E0 Infer already sits near this floor,
  so reference accuracy is near the ceiling this geometry allows and is easily broken by a bad
  ID6 prediction.
- **Cardiac axis is segmentation-limited, not floor-limited:** Gold Card (2.99°) ≥ E0 Infer
  Card (2.98°); E2 improves the Infer card axis toward Gold on hard cases.
- **Hard cases reveal prediction-shape error:** Gold Full 9.41° vs E0 Infer 14.18° → ~5° of the
  hard-case error is prediction shape; E2 brings it to 12.19° (closer to the Gold floor).
(`metrics/gold_infer_comparison.csv`.)

## 9. The 64 hard cases (E0 Full Error ≥ 10°)

- Mean Full 14.18° → 12.19° (Δ −1.99°); median 12.90° → 10.80°; 46/64 improved (sign).
- Card 11.04° → 9.96° (Δ −1.08°); Ref 6.47° → 5.95° (Δ −0.52°). **Both axes improve**, cardiac
  more; ID4/ID5 OBB-vs-Gold errors drop (id4 8.34→7.88, id5 8.67→7.41).
- E2's clearest benefit: it rescues a large share of E0's difficult cases, predominantly via the
  cardiac axis. P95 of this set rises because a few hard cases worsen sharply (geometry
  outliers), while the typical hard case improves.

## 10. The 5 severe negative-transfer cases (anonymized)

All are external Multicenter Ne; ΔFull +29.6° to +43.6°; kept in all formal totals.

| Slot | Full E0→E2 | Card E0→E2 | Ref E0→E2 | ID6 centroid shift |
|---|---|---|---:|---:|
| EXT-W01 | 4.1 → 47.6 | 0.3 → 0.0 | 3.8 → 47.6 | ~3 → ~590 px |
| EXT-W02 | 7.5 → 48.5 | 6.4 → 2.0 | 1.1 → 46.4 | ~17 → ~619 px |
| EXT-W03 | 0.5 → 35.8 | 4.1 → 3.4 | 4.5 → 39.2 | ~6 → ~644 px |
| EXT-W04 | 3.6 → 35.1 | 3.9 → 3.1 | 0.3 → 32.1 | ~8 → ~590 px |
| EXT-W05 | 1.0 → 30.6 | 0.9 → 1.6 | 2.0 → 29.0 | ~5 → ~629 px |

**Mechanism (verified on masks, no per-case identifiers here):** in every case the cardiac axis
is fine or better, but the reference axis collapses. The reference line is ID6-centroid →
ID1-centroid; ID1's centroid barely moves, while E2's **ID6 largest-connected-component flips
from the correct region to a large spurious component**, moving the ID6 centroid ~590–644 px and
rotating the reference axis by tens of degrees. The Gold reference axis in these cases is
accurate (1.5–2.6°) with the correct ID6 component → this is an **ID6 prediction-morphology
error**, not a geometry-definition or GT problem. ID6 Dice drops only modestly (0.60–0.75 →
0.45–0.63), so **Dice is blind to this topological/centroid failure**. Across the full cohort, 7
cases (6 external + 1 internal, all Ne) show an E2 ID6-centroid jump > 100 px; the 5 above are
the most extreme.

## 11. Dice ↔ angle decoupling

- Correlation between ΔDice (mean) and ΔFullError: internal r ≈ −0.10, external r ≈ −0.08 —
  essentially none. Many cases show opposite-sign changes (Dice down / angle up and vice versa).
- ID6 Dice change vs Reference error change: Pearson −0.35 (all) / −0.51 (external), **but after
  removing the 5 severe cases the Pearson collapses to ~0 and Spearman is ≈ −0.07 everywhere**.
  So the apparent relationship is **outlier-driven**; ID6 Dice is a poor monitor of the
  reference-axis failure, whereas centroid / largest-component is diagnostic.
- Implication: segmentation similarity does not predict measurement change; any transfer benefit
  flows through orientation, not voxel overlap.

## 12. Limitations and neutral scientific conclusion

1. **Neutral result on the clinical endpoint.** Cross-trimester pretraining did not improve early
   trimester Full Angle at the population level; Ne and external deltas are within ±0.25° and not
   significant; P95 changes are inconsistent in sign.
2. **Segmentation unchanged** (Dice/IoU ≤0.004; HD95 ≤1 px) — clean transfer but no early gain.
3. **Po results are suggestive only** (tiny n, wide CIs).
4. **Mechanistic, distributional finding:** E2 helps the Cardiac axis (especially hard / Po
   cases) but harms the Reference axis, with rare severe failures driven by ID6 centroid /
   largest-component flips in Multicenter normal cases. All formal averages retain the 5 severe
   cases; removing them flips external ΔMAE from +0.12° to −0.09°.
5. **Scope:** 2D only, single dataset, single pretrained checkpoint, fixed geometry. No claim of
   causal mechanism; the ID6 domain-shift reading is a hypothesis for future work (e.g.
   reference-axis-robust training, centroid/LCC-aware ID6 handling, partial-layer transfer).
6. **Conclusion:** with the same architecture, reinitialized 7-class head, and frozen E0
   folds/plans, naive mid→early weight transfer does **not** beat the E0 scratch baseline on the
   clinical endpoint — a null/neutral result, not an improvement.

Case-level tables, prediction masks, and weights remain server-side and are intentionally not
archived.
