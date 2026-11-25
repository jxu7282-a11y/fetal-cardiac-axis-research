# E1-pre — Mid-Trimester Full-Data nnUNetv2 2D Pretraining

**Question:** Can a standard nnUNetv2 2D model be trained on all usable mid-trimester
four-chamber ultrasound cases, with a network compatible with the early-trimester E0
plans, to provide a checkpoint for later early-trimester fine-tuning?

**Answer (short):** Yes. After deterministic image-level deduplication, 890/890 usable
cases were trained for 300/300 epochs in 2D with `nnUNetPlansFrom501`. The resulting
network matches E0 in all 382 non-segmentation keys and requires only reinitialization
of the 14 segmentation-output keys. The reported same-data Dice of 0.9784 is training
monitoring only and says nothing about held-out or external generalization.

---

## 1. Scope and experimental setup

| Item | Value |
|---|---|
| Experiment | E1-pre mid-trimester pretraining |
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

No post-training fine-tuning, evaluation on early-trimester cases, or model training was
started as part of this archive task.

## 2. Data audit and deduplication

The source contained 905 RGB PNG images, 905 single-channel PNG masks, and 905
annotation-info JSON files. Images, masks, and annotation files matched by reliable case
stem in 905/905 cases. In the retained set, image and mask dimensions matched in 890/890
cases. No empty masks or corrupt files were found. PNG images and masks carry no DICOM/affine spatial metadata, so orientation checks were limited to size, decoding, and same-source consistency; no rotation, flip, or non-isotropic scaling was applied.

Exact image-file SHA256 identified **15 duplicate groups, each containing 2 files**
(30 files total). One case per group was retained using the deterministic rule
**lexicographically smallest case ID**. The annotation files contained only geometric
fields (`bbox`, `landmark`, `roi_box`) and no review, approval, version, or final-status
field that could support a content-based preference.

The duplicate masks were **not identical**. The differing-pixel count ranged from 6,022
to 46,473 (3.9%–12.7% of the image) across the 15 groups. The retained case kept its own
image, mask, and annotation file. No masks were merged, substituted, or selected by Dice
or area. This rule is deterministic; it does not establish which annotation is clinically
more correct.

Additional audit note: `roi_box` (all 905 cases) and at least one `landmark` (98/905 cases) fall outside the image bounds; this coordinate anomaly was recorded but not used for training, cropping, or mask rewriting.

| Count | Value |
|---|---:|
| Original images | 905 |
| Unique image SHA256 values after deterministic selection | 890 |
| Duplicate groups | 15 |
| Files in duplicate groups | 30 |
| Retained cases | 890 |
| Excluded duplicate cases | 15 |
| Image/mask match in retained set | 890/890 |
| Empty masks | 0 |
| Corrupt files | 0 |

## 3. Label handling

The masks use integer label IDs **0–9**, ten classes in total. Every retained case
contains all ten IDs. ID 0 is `background` under the nnU-Net convention. No authoritative
mid-trimester label dictionary or label-name field was available.

Therefore:

- IDs 1–9 remain **neutral placeholders** `label_1`…`label_9`;
- no anatomical name was assigned or inferred;
- no label was deleted, merged, or renumbered;
- the original-to-nnU-Net mapping is reversible identity `0→0 … 9→9`;
- no mid-trimester label was equated with an early-trimester label.

Aggregate label totals for the 890 retained masks are recorded in
`metrics/data_audit_summary.csv`. These totals describe class prevalence only; they do
not assign semantics.

## 4. Plans transfer and E0 compatibility

The early-trimester E0 plans were treated as the target network design. An independent
mid-trimester plans file was created with the official mechanism:

```text
nnUNetv2_move_plans_between_datasets \
  -s 501 -t 503 -sp nnUNetPlans -tp nnUNetPlansFrom501
```

This produced `nnUNetPlansFrom501`; the E0 plans file was not edited. Image channels
(3), patch size (768 x 768), batch size (5), and the 8-stage `PlainConvUNet` topology all
match E0. Mid-trimester intensity statistics were recomputed rather than inherited from
E0, and the data were preprocessed with those statistics. In preprocessed arrays, a `-1` value outside the non-zero cropping box is nnU-Net's ignore value, not an extra source-mask class.

The prepared dataset `Dataset503_MidPregnancyCardiac` has `imagesTr` 890, `labelsTr` 890, and `dataset.json` `numTraining=890`. It uses the 2D configuration with the `NaturalImage2DIO` reader/writer, 3-channel RGB inputs and single-channel PNG labels. The official `verify_dataset_integrity` check passed; after re-preprocessing, 890/890 image, segmentation, and `.pkl` files exist, and cropped-region masks are pixel-identical to the source masks.

| Compatibility check | Result |
|---|---|
| E0 plans SHA256 unchanged | yes (`3f494bf3…d013f0`) |
| Input channels | 3 vs 3 |
| Patch / batch | 768×768 / 5 in both |
| Non-segmentation keys | 382 vs 382 |
| Missing / extra non-seg keys | 0 / 0 |
| Non-seg key shape mismatches | 0 |
| Segmentation output keys | 14 (`seg_layers`) |
| Output classes | E0: 7; mid: 10 |

The official nnU-Net 2.8.1 `load_pretrained_weights` loader transfers matching
non-segmentation keys and **skips `.seg_layers.`**. It therefore can load the
mid-pretrained checkpoint into an early-trimester network, while the early-trimester
segmentation output head is reinitialized for its own class count. No output-class
semantics are shared merely because IDs coincide.

## 5. Training execution and monitoring

The run completed **300/300 epochs** and exited normally. The mean epoch duration was
33.71 seconds; the sum of the 300 epoch timers was 2.81 hours. The final epoch recorded
train loss -0.9127 and validation loss -0.9386; the best same-data monitoring validation loss was -0.9387. `checkpoint_final.pth` was saved on `on_train_end`; the periodic `checkpoint_latest.pth` (written every 50 epochs) is auto-removed after the final save, so only `final` and `best` checkpoints remain.

`fold=all` in nnU-Net sets the training and validation key lists to the same full
collection of 890 cases. It is therefore an **in-sample monitoring pass**, not an
independent validation or test set.

| Monitoring metric (in-sample) | Value |
|---|---:|
| Cases | 890 (all also trained on) |
| Mean foreground Dice | 0.9784 |
| Mean foreground IoU | 0.9581 |
| label_1 Dice | 0.9799 |
| label_2 Dice | 0.9829 |
| label_3 Dice | 0.9809 |
| label_4 Dice | 0.9832 |
| label_5 Dice | 0.9808 |
| label_6 Dice | 0.9853 |
| label_7 Dice | 0.9619 |
| label_8 Dice | 0.9844 |
| label_9 Dice | 0.9659 |

**These values are not generalization performance.** They must not be presented as
mid-trimester held-out accuracy or as clinical evidence. The intended evaluation of
transfer remains the early-trimester E0/E1 comparison using the frozen formal folds and
external/prospective cohorts.

## 6. Artifacts and integrity

The model weight is retained on the training server only:

```text
checkpoint_final.pth
SHA256 df94a1d9aac1557a50d63856d684ef236508b45731dbcfd3479c85493314cd9f
```

`checkpoint_best.pth` is also retained server-side. The training logs, debug metadata,
same-data validation outputs, dataset plans, dataset JSON, and experiment configuration
remain under the local experiment directory.

Weights, predictions, logs, original images, masks, annotations, case identifiers, and
case-level manifests are **not** included in this Git archive. The files here are limited
to aggregate counts, configuration, rules, and monitoring metrics.

## 7. Limitations and open issues

1. The anatomical semantics and boundary definitions of labels 1–9 remain unconfirmed;
   they remain neutral IDs.
2. The 15 excluded duplicate cases have non-identical masks. Deterministic selection is
   not evidence that the retained annotation is clinically superior. The original files
   remain untouched on the server.
3. The annotation geometry is inconsistent with the current PNG pixel space for some
   records; those coordinates were not used for image or mask transformation.
4. `fold=all` provides no held-out validation or test measurement.
5. E0 and mid-trimester output classes are not equivalent. Only 382 non-segmentation
   parameters are transferable; the 14 `.seg_layers.` keys must be reinitialized.
6. Whether mid-trimester pretraining improves early-trimester cardiac-axis measurement
   is still an untested hypothesis. No E1 fine-tuning result is claimed here.

## 8. Reproduction note

This archive alone is not runnable. Reproduction requires the private mid-trimester
dataset, nnU-Net v2 2.8.1, the local plans file, and a GPU. The local preparation scripts
and patient-level manifests are intentionally excluded from Git.
