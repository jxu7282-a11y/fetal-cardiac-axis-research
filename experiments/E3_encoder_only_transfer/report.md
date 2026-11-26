# E3 — Mid-Trimester Encoder-Only Transfer to Early-Trimester Cardiac-Axis Segmentation

**Version:** `e3-encoder-only-v1.0`  
**Archived:** 2026-11-26  
**Status:** training complete (5/5 folds × 300 epochs), evaluation and statistics complete

---

## 1. Hypothesis and position in the project

E2 asked whether transferring the *whole* mid-trimester network helps early-trimester
cardiac-axis measurement; the answer was a null result. E3 isolates the encoder:

> **H-E3:** the mid-trimester **encoder** alone carries the transferable anatomical
> representation; a fresh decoder and fresh segmentation heads will adapt it to the
> early-trimester task better than a wholesale transfer.

E3 is therefore a controlled ablation of E2 — same data, folds, plans, schedule, and
evaluation protocol; the only change is *which* parameters start from mid-trimester
weights.

## 2. Data

| Cohort | N | Ne / Po | Role |
|---|---:|---|---|
| Retrospective | 621 | 571 / 50 | five-fold development, OOF evaluation |
| Multicenter | 442 | 434 / 8 | external test |
| Prospective | 411 | 400 / 11 | prospective test |
| **Total** | **1474** | 1405 / 69 | |

- Splits are the frozen E0 `fold_v2.csv` (not regenerated); `splits_final.json` is the E0
  version.
- Ground truth is **`label_gold`** only; `label_infer` is never used.
- External and prospective cohorts were never used for training, epoch selection,
  hyper-parameter choice, or post-processing tuning.

## 3. Method — encoder-only initialisation

Source checkpoint: `local/mid_pretraining/results/checkpoint_final.pth`,
SHA256 `df94a1d9aac1557a50d63856d684ef236508b45731dbcfd3479c85493314cd9f`
(371,132,749 B, epoch 300, identifier re-verified before every fold).

| Network group | Params | E3 initialisation |
|---|---:|---|
| `encoder.*` (64 unique tensors, 256 state_dict entries) | 18,881,184 | **loaded from mid-trimester** |
| `decoder.stages.*` | 23,595,840 | fresh random |
| `decoder.transpconvs.*` (`ConvTranspose2d`, has weights) | 3,844,064 | fresh random |
| `decoder.seg_layers.*` (7 heads × 2) | 14,161 | fresh random |
| **Total** | **46,335,249** | |

Three architecture facts made this non-trivial and were verified rather than assumed:

1. `net.decoder.encoder is net.encoder` — the encoder object is shared into the decoder
   graph, so each encoder tensor appears under `encoder.*`, `decoder.encoder.*`, and a
   second alias inside its own conv block (148 unique parameter objects vs 396
   `state_dict` entries).
2. The official loader (`-pretrained_weights`) **cannot** be used: it asserts that every
   non-`.seg_layers.` key exists in the checkpoint, so an encoder-only state dict is
   rejected.
3. `ConvTranspose2d` weights live under `decoder.transpconvs.*` and belong to the decoder;
   excluding them is required for a true encoder-only transfer.

Implementation (`src/nnunet_ext/nnUNetTrainer_E3EncoderOnly.py`) copies the 64 encoder
parameter objects in place (avoiding `load_state_dict` alias-ordering effects), then
asserts:

- all 256 encoder `state_dict` entries equal the checkpoint bit-for-bit;
- all 140 decoder/seg entries are bit-identical to their pre-transfer snapshot;
- **zero** decoder/seg entries equal their checkpoint value (counter-proof that no
  decoder weight leaked in).

Any precondition failure aborts training instead of continuing. Encoder weights are **not
frozen** — all parameters are fine-tuned. Per-fold audits are
`encoder_load_fold_{0..4}.json`; the aggregate audit is `encoder_loading_audit.md`.
No optimizer state, epoch counter, or E0/E2 weight is reused; each fold starts at epoch 0
from the same mid-trimester checkpoint and folds never inherit from each other.

## 4. Training configuration

| Item | Value |
|---|---|
| nnU-Net version | 2.8.1 (PyTorch 2.7.1+cu118) |
| Task / configuration | `Dataset501_EarlyRetroCardiac`, 2D |
| Architecture | `PlainConvUNet`, 8 stages, 3 input channels, 7 output heads |
| Trainer | `nnUNetTrainer_E3EncoderOnly` (subclass of stock `nnUNetTrainer`) |
| Epochs | 300 per fold |
| Folds | 0, 1, 2, 3, 4 (serial) |
| Loss / optimizer / LR / batch / patch / augmentation / preprocessing / plans | stock nnU-Net defaults, unchanged |
| Patch size / batch size | 768×768 / 5 |
| GPU | one physical GPU (3), shared, non-interfering |

Verification performed before training: a CPU check of the real network against the real
checkpoint, then a GPU smoke run through nnU-Net's own trainer lookup (one shortened epoch,
2 train steps + 1 validation step, `train_loss 2.2807` / `val_loss 2.1832`).

Training outcome:

| Fold | Epochs | Exit | Best EMA Dice | Final val Dice |
|---|---:|---:|---:|---:|
| 0 | 300 | 0 | 0.9371 | 0.9359 |
| 1 | 300 | 0 | 0.9345 | 0.9324 |
| 2 | 300 | 0 | 0.9361 | 0.9343 |
| 3 | 300 | 0 | 0.9372 | 0.9362 |
| 4 | 300 | 0 | 0.9407 | 0.9306 |

All checkpoints readable (396 keys each, `final` at epoch 300). SHA256 per fold in
`checkpoints_sha256/`. Weights stay on the server; only hashes are archived.

An earlier pilot run (2026-11-25 15:08 start) was interrupted and is archived separately as
`local/e3_encoder_only_transfer/pilot_aborted_20261125_1508/`. It is **not** part of E3.

## 5. Evaluation protocol

Identical to E0/E2, without modification:

- GT: `local/phase0/fixed_gt_v1.csv` (frozen)
- Folds: `local/phase0/fold_v2.csv`
- Geometry: `src/cardiac_axis/evaluate.py` (ID4/ID5 OBB-weighted card axis; ID6→ID1 centroid
  reference axis)
- Internal: each fold predicts its own validation set, pooled OOF over 621
- External: 5-fold ensemble of `checkpoint_best.pth`, `--disable_tta`
- Metrics: Dice, IoU, HD95 per label; Full Angle MAE/MedAE/P90/P95; Card/Reference axis error;
  measurement failure rate
- Statistics: paired by case `uid`, 10000 bootstrap resamples, 95% percentile CI, seed 42

## 6. Results

### 6.1 Internal OOF (621)

| Metric | E0 | E2 | **E3** |
|---|---:|---:|---:|
| Full MAE (°) | 3.639 | 3.607 | **3.647** |
| Full MedAE (°) | 2.665 | 2.662 | **2.806** |
| Full P90 (°) | 7.606 | 7.457 | **7.176** |
| Full P95 (°) | 9.924 | 9.460 | **9.173** |
| Card MAE (°) | 3.088 | 3.112 | **3.188** |
| Reference MAE (°) | 2.575 | 2.458 | **2.404** |
| Dice | 0.9354 | 0.9337 | **0.9339** |
| IoU | 0.8914 | 0.8878 | **0.8882** |
| HD95 (px) | — | — | **10.55** |

E3 reports 620 valid Full Angle values, not 621: one case is a genuine measurement failure
(section 7).

### 6.2 External (853)

| Metric | E0 | E2 | **E3** |
|---|---:|---:|---:|
| Full MAE (°) | 3.740 | 3.860 | **3.693** |
| Full MedAE (°) | 3.077 | 3.048 | **3.004** |
| Full P90 (°) | 7.700 | 7.664 | **7.342** |
| Full P95 (°) | 9.351 | 9.506 | **8.848** |
| Card MAE (°) | 2.906 | 2.704 | **2.768** |
| Reference MAE (°) | 2.701 | 3.085 | **2.922** |
| Dice | 0.9295 | 0.9275 | **0.9282** |
| IoU | 0.8775 | 0.8735 | **0.8746** |
| HD95 (px) | — | — | **16.80** |

By cohort (E3): **Multicenter 442 — Full MAE 4.030°, Dice 0.9262, IoU 0.8731.
Prospective 411 — Full MAE 3.330°, Dice 0.9303, IoU 0.8763.**

### 6.3 Paired bootstrap (E3 − comparator, Full Angle)

| Scope | Subset | Comparison | n | Δ mean | 95% CI | Significant |
|---|---|---|---:|---:|---|---|
| Internal | ALL | E3 vs E0 | 620 | +0.013° | [−0.156, +0.195] | no |
| Internal | ALL | E3 vs E2 | 620 | +0.041° | [−0.081, +0.169] | no |
| Internal | Ne | E3 vs E0 | 570 | +0.123° | [−0.030, +0.286] | no |
| Internal | **Po** | **E3 vs E0** | **50** | **−1.233°** | **[−2.375, −0.055]** | **yes (better)** |
| Internal | Po | E3 vs E2 | 50 | −0.170° | [−0.907, +0.597] | no |
| External | ALL | E3 vs E0 | 853 | −0.047° | [−0.226, +0.161] | no |
| External | **ALL** | **E3 vs E2** | **853** | **−0.167°** | **[−0.316, −0.032]** | **yes (better)** |
| External | **Multicenter** | **E3 vs E2** | **442** | **−0.287°** | **[−0.561, −0.049]** | **yes (better)** |
| External | Prospective | E3 vs E2 | 411 | −0.038° | [−0.143, +0.064] | no |

Axis-level (ALL, paired): internal E3 vs E0 improves Reference (−0.175° [−0.406, −0.030])
while Card is non-significant. Externally the Reference axis **worsens** versus E0
(+0.221° [+0.004, +0.498]). Card axis changes are non-significant in both directions.

### 6.4 Error burden and per-case movement

Full Angle ≥10°: internal E0 31, E2 26, **E3 27**; external E0 33, E2 37, **E3 24 (2.81%)**.
≥15°: internal E0 14, E2 11, E3 10; external 6 / 8 / 6.

Per-case change at |Δ| ≥ 1°: internal vs E0 109 improved / 111 worsened; vs E2 110 / 111.
External vs E0 178 improved / 135 worsened; vs E2 161 improved / 124 worsened.

### 6.5 Cohort matching audit

All three models share identical uid sets (621 internal, 853 external), zero duplicate uids,
zero missing predictions, and bit-identical GT values (max difference 0.0). No case was
silently dropped anywhere in this analysis.

## 7. The one measurement failure

`case_hash` recorded in `analysis/failure_cases_anonymized.csv`; the real uid is kept only
in `local/`.

- Fold 0, internal, Ne. `valid_pred = True`, Card axis computed (1.66° error), **Reference
  axis NaN**.
- Root cause verified on the masks: the E3 prediction contains labels {0,2,3,4,5,6} and
  **zero pixels of label 1**. Gold has 1804 ID1 pixels; E0 predicted 229.
- Since the Reference axis is built from ID6→ID1 centroid, a complete ID1 dropout makes the
  Reference axis and the Full Angle undefined.

This is a real segmentation failure, not an evaluation bug, and it is reported rather than
dropped. It affects only that case's Reference/Full values; the other 620 internal cases are
unaffected.

## 8. Scientific conclusion

1. **Segmentation is unchanged.** Dice/IoU differ from E0 by ≲0.002 in every cohort, well
   inside training variance. Encoder-only transfer did not improve the intermediate task.
2. **No overall clinical advantage over E0.** Internal Δ = +0.013° and external Δ = −0.047°,
   both with CIs spanning zero. E3 cannot be claimed as better than the scratch baseline.
3. **E3 is significantly better than E2 externally** (−0.167°, CI [−0.316, −0.032]), driven
   by Multicenter (−0.287°). Since E2 transfers the whole network and E3 only the encoder,
   this is evidence that wholesale decoder transfer was harmful on the external set.
4. **Benefits are subgroup-localised**: internal pathological cases improve strongly vs E0
   (−1.233° on n = 50), and the external tail tightens (≥10° errors 24 vs 33/37).
5. **The Reference axis moves inconsistently**: better internally, worse externally versus
   E0. This is a cohort-dependent change, not a uniform improvement, and it partly offsets
   the Full Angle gains.
6. The single total ID1 dropout shows the encoder-only model can lose a thin reference
   structure outright; such failures need explicit reporting in any deployment claim.

**Bottom line:** encoder-only transfer does not beat the E0 scratch baseline on overall
clinical measurement, but it does beat the E2 full-network transfer externally, with gains
concentrated in pathological internal cases and multicenter external cases, alongside a
cohort-dependent Reference-axis trade-off.

## 9. Limitations

- One seed, one architecture, one dataset; no repeated-run variance estimate.
- Internal Po is n = 50 and external Po is n = 19 — subgroup effects are exploratory.
- Nine pairwise subgroup tests were run without multiplicity correction; CIs are
  per-comparison and should be read as such.
- The two image sets (manual-measurement vs segmentation) differ in cropping/scaling; GT
  comparison is angle-domain only, as fixed by the protocol. Any residual rotation in
  preprocessing is an unquantified risk shared by E0/E2/E3.
- Encoder transfer is verified bit-exactly, but *which* encoder property transfers is not
  explained; no representation analysis was performed.
- HD95 is reported as a mean over labels; per-label values are in the local artifacts.

## 10. Where the artifacts live

Archived in Git (this directory): report, config, summary JSON, anonymised case-level CSVs,
statistical tables, SHA256 manifest, evaluation scripts.

Server-only, never uploaded: model weights (~1.85 GB per fold × 10 files), external
prediction masks (853 PNGs), raw case-level CSVs with real uids, training logs, and the
aborted pilot. Paths:

- training results: `local/e3_encoder_only_transfer/results_nnunet/`
- evaluation: `local/e3_encoder_only_transfer/evaluation/`
- analysis: `local/e3_encoder_only_transfer/analysis/`
- logs: `local/e3_encoder_only_transfer/logs/`
