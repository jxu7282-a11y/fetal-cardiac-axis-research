# Mid-Pretraining — Mid-Trimester Full-Data nnUNetv2 2D Pretraining

Version: `mid-pretraining-v1.0`
Status: complete (1/1 run, 300/300 epochs)

## What this is

A standard nnUNetv2 2D segmentation model trained on **all deduplicated mid-trimester**
fetal four-chamber ultrasound cases. It exists to provide **transferable visual features**
for a later early-trimester fine-tuning experiment (E2), not to establish mid-trimester
clinical performance.

The run used `nnUNetPlansFrom501`, a plans identifier created from the early-trimester E0
plans through nnU-Net's official plans-transfer mechanism. The E0 dataset, split, plans,
code, and model were **not modified**.

> Note: this pretraining run was previously summarised in the repository as
> `experiments/E1_mid_pretraining/`. This directory (`experiments/Mid_pretraining/`) is the
> consolidated, self-contained archive of the same completed run and its data audit; it does
> not describe a new experiment.

## Contents

| File | Content |
|---|---|
| `README.md` | this file |
| `report.md` | data audit, deduplication, label handling, training, in-sample monitoring, E0 compatibility, open issues |
| `config.json` | configuration actually used: dataset counts, deduplication rule, plans, trainer, epochs, fold, network, weight checksum |
| `metrics/data_audit_summary.csv` | group-level counts, label-level aggregate pixel totals, and same-data monitoring metrics per label |

## Headline numbers

| Item | Value |
|---|---:|
| Original cases | 905 |
| SHA256 duplicate groups | 15 groups / 30 files |
| Deduplicated cases | 890 |
| Valid training cases | 890 |
| Image/mask match | 890/890 |
| Label IDs | 0–9 (10 classes; semantics 1–9 unconfirmed) |
| Epochs completed | 300/300 |
| Configuration / Fold | 2d / `all` |
| Same-data monitoring Dice | 0.9784 |
| Same-data monitoring IoU | 0.9581 |
| Final weight SHA256 | `df94a1d9aac1557a50d63856d684ef236508b45731dbcfd3479c85493314cd9f` |

Dice 0.9784 is **in-sample training monitoring only**. Under `fold=all`, every monitored
case is also a training case; this is **not** a held-out or generalization metric.

## E0 compatibility

- 382 non-segmentation state-dict keys match E0 by name and shape.
- 14 `.seg_layers.` output keys differ because E0 has 7 classes and this model has 10.
- The official nnU-Net 2.8.1 `-pretrained_weights` loader transfers the 382 matching keys,
  skips `.seg_layers.`, and reinitializes the early-trimester output head.

## Not in this repository

Deliberately excluded (kept on the training server only, under `local/mid_pretraining/`):

- original ultrasound images, masks, annotation files, and TAR archives;
- any case-level manifest, case-identifier list, or case-to-file mapping;
- training logs, debug dumps, predictions, and validation images;
- model weights (`checkpoint_final.pth`, `checkpoint_best.pth`).

The final weight remains outside Git. Its SHA256 is recorded in `config.json` and
`report.md` for provenance. Reproduction requires a GPU and the private dataset; this
archive alone is not runnable.
