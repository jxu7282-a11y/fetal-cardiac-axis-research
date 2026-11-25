# Mid-Pretraining — Mid-Trimester Full-Data nnUNetv2 2D Pretraining

**Question:** Can a standard nnUNetv2 2D model be trained on all usable mid-trimester
four-chamber ultrasound cases, with a network compatible with the early-trimester E0
plans, to provide a checkpoint for later early-trimester fine-tuning?

**Answer (short):** Yes. After deterministic image-level deduplication, 890/890 usable cases
were trained for 300/300 epochs in 2D with `nnUNetPlansFrom501`. The resulting network matches
E0 in all 382 non-segmentation keys and requires only reinitialization of the 14
segmentation-output keys. The reported same-data Dice of 0.9784 is **training monitoring only**
and says nothing about held-out or external generalization.

---

## 1. Scope and experimental setup

| Item | Value |
|---|---|
| Experiment | Mid-trimester pretraining |
| Dataset | `Dataset503_MidPregnancyCardiac` |
| Task | 2D multi-label segmentation |
| Training | scratch, random initialization; no ImageNet or external pretrained weights |
| Trainer | `nnUNetTrainer_300Epochs` = stock nnUNetTrainer with `num_epochs=300` only |
| Plans | `nnUNetPlansFrom501`, derived from E0 `Dataset501/nnUNetPlans` |
| Network | `PlainConvUNet`, 8 stages, 46,341,318 parameters, 3 inputs, 10 outputs |
| Batch / patch | 5 / 768 x 768 |
| Loss / optimizer | nnUNet default DC+CE with deep supervision / SGD + poly LR |
| Epochs | 300/300 |
| Fold | `all` |
| GPU | physical GPU 3, NVIDIA H20-3e; visible internally as `cuda:0` |

No fine-tuning, early-trimester evaluation, or additional training was started as part of
this archive task.

## 2. Data audit and deduplication

The private source contained 905 RGB PNG images, 905 single-channel PNG masks, and 905
annotation-info JSON files. Images, masks, and annotation files matched by reliable case
stem in 905/905 cases. In the retained set, image and mask dimensions matched in 890/890
cases. No empty masks or corrupt files were found. PNG files carry no DICOM/affine spatial
metadata, so orientation checks were limited to size, decoding, and same-source
consistency; no rotation, flip, or non-isotropic scaling was applied.

Exact image-file SHA256 identified **15 duplicate groups, each containing 2 files**
(30 files total). One case per group was retained using the deterministic rule
**lexicographically smallest case ID**. The annotation files contained only geometric
fields (`bbox`, `landmark`, `roi_box`) and no review, approval, version, or final-status
field that could support a content-based preference.

The duplicate masks were **not identical**. The differing-pixel count ranged from 6,022 to
46,473 (3.9%–12.7% of the image) across the 15 groups. The retained case kept its own image,
mask, and annotation file. No masks were merged, substituted, or selected by Dice or area.
**This is a deterministic deduplication rule only; it does not claim the retained mask is
clinically more correct.**

Additional audit notes: `roi_box` (all 905 cases) and at least one `landmark` (98/905 cases)
fall outside the image bounds; this coordinate anomaly was recorded but not used for
training, cropping, or mask rewriting.

## 3. Labels

- Original Label IDs are `0–9` (10 classes). Every retained case contains all 10 IDs.
- In `dataset.json` they are named `background` (0) and neutral placeholders `label_1` …
  `label_9` (1–9). **No anatomical meaning is assigned to IDs 1–9.** No authoritative
  mid-trimester label dictionary was found.
- The original-to-nnU-Net label mapping is a reversible identity (0→0, 1→1, …, 9→9).
- Aggregate label pixel totals and per-ID case counts are in
  `metrics/data_audit_summary.csv`.
- **Do not equate any mid-trimester label with an early-trimester E0 label.** Cross-trimester
  transfer moves network feature weights only and reinitializes the output head.

## 4. nnU-Net data and preprocessing

- Independent dataset `Dataset503_MidPregnancyCardiac`: `imagesTr` 890, `labelsTr` 890,
  `dataset.json` `numTraining=890`.
- Official `verify_dataset_integrity`: PASS; `NaturalImage2DIO`; 3-channel RGB, single-channel
  PNG labels.
- Preprocessing used `nnUNetPlansFrom501`, produced by the official
  `nnUNetv2_move_plans_between_datasets` from E0 `Dataset501/nnUNetPlans` (configuration `2d`;
  patch 768×768; batch 5; 8-stage `PlainConvUNet`).
- The mid-trimester intensity fingerprint (Z-score statistics) was **recomputed from the
  mid-trimester dataset**, not inherited from E0. After re-preprocessing, 890/890 image, seg,
  and pkl files exist; cropped-region masks are pixel-identical to source masks.
- A `-1` value appears in preprocessed regions outside the non-zero cropping box; this is
  nnU-Net's ignore value, not an extra source-mask class.

## 5. Training progress

- 300/300 epochs completed; mean epoch time 33.71 s; total epoch time ≈ 2.81 h.
- Final train loss −0.9127; final val loss −0.9386; best (same-data monitoring) val loss −0.9387.
- `checkpoint_final.pth` was saved on `on_train_end`; the periodic `checkpoint_latest.pth`
  (every 50 epochs) is auto-removed after final save, so only `final` and `best` remain.

## 6. Same-data monitoring (NOT an independent-validation result)

| Item | Value |
|---|---:|
| Monitoring cases | 890 (same cases as training) |
| Mean foreground Dice | 0.9784 |
| Mean foreground IoU | 0.9581 |
| label_1 … label_9 Dice | 0.9799, 0.9829, 0.9809, 0.9832, 0.9808, 0.9853, 0.9619, 0.9844, 0.9659 |

**Critical interpretation boundary.** Under `fold=all`, every "validation" case is also a
training case. These numbers are **in-sample training monitoring only** and **must not** be
reported as held-out, external, prospective, or generalization performance. Independent
early-trimester judgement rests on the E0 five-fold OOF baseline and the E2 fine-tuning
evaluation, not on this quantity.

## 7. E0 compatibility

- E0 (early-trimester `Dataset501`, 7 classes) and this model (10 classes) share **382
  non-segmentation keys** with identical name and shape.
- The only differences are the **14 `.seg_layers.`** output keys (E0 7 classes vs mid 10).
- The official nnU-Net 2.8.1 `-pretrained_weights` loader transfers the 382 matching keys,
  skips `.seg_layers.`, and reinitializes the early-trimester segmentation head.
- E0 plans, data, folds, and model were not modified. The E0 plans SHA256 is
  `3f494bf38c14e3add04a4063817df9dbf8ff369d18acce483513c91d7dc013f0`.

## 8. Open issues

1. The authoritative anatomical names and boundary definitions of mid-trimester labels 1–9
   remain unconfirmed; only neutral placeholder names are used.
2. The clinical correctness of the 15 duplicate-image masks cannot be determined from the
   available annotation info; the exclusion is deterministic, not clinical.
3. `roi_box` / some `landmark` coordinates are inconsistent with PNG pixel space and were not
   used for transformation or geometric supervision.
4. `fold=all` uses all 890 cases for both training and same-data monitoring; it is not an
   independent generalization evaluation.
5. E0 is only a network-topology compatibility target; the 10-class mid head cannot share the
   7-class E0 output head, so `.seg_layers.` must be skipped and reinitialized on transfer.

## 9. Artifacts and integrity

Deliberately **not** in Git (server-only under `local/mid_pretraining/`): original images,
masks, annotation files, TAR archives, case manifests / mappings, training logs, debug dumps,
predictions, and both `checkpoint_final.pth` / `checkpoint_best.pth`.

Final weight SHA256 (for provenance):

```
df94a1d9aac1557a50d63856d684ef236508b45731dbcfd3479c85493314cd9f
```

## 10. Conclusion

A mid-trimester 2D nnUNetv2 model was trained on all 890 deduplicated cases for 300 epochs
with an E0-derived plans (`nnUNetPlansFrom501`) and is structurally compatible with the E0
early-trimester network (382 transferable keys, 14 reinitialized output keys). This archive
provides the pretrained checkpoint provenance for the follow-up E2 fine-tuning study; it makes
**no** mid-trimester clinical-performance claim.
