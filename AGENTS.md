# Project Instructions: Fetal Cardiac Axis Research

## 1. Project Context

This project studies whether mid-trimester fetal cardiac ultrasound
segmentation data can improve early-trimester cardiac axis angle
measurement through pretraining and fine-tuning.

The final clinical objective is accurate early-trimester cardiac
axis angle measurement, not segmentation performance alone.

## 2. Mandatory Documents

Before performing any project task, read:

- docs/project/00_PROJECT_MASTER.md
- docs/project/01_DATA_SPEC.md
- docs/project/02_GT_PROTOCOL.md
- docs/project/03_EXPERIMENT_LOG.md
- docs/project/04_DECISION_LOG.md
- docs/project/05_OPEN_QUESTIONS.md
- docs/original_specs/CARDIAC_AXIS_DATA_READING_SPEC_v1.md
- docs/original_specs/CARDIAC_AXIS_METHOD_HISTORY.md

If the original data contract is available, also read:
cardiac_axis_data_contract_v1.yaml.

If a required document is missing, report it. Do not invent its contents.

## 3. Data Safety

- Original dataset directory is /data5/xjx/zaoyun/data/j.
- Treat all original data as READ-ONLY.
- Never delete, rename, overwrite, move, or modify original images,
  TAR files, masks, manifests, or fold files.
- Never upload patient images, TAR annotations, identifiable manifests,
  or patient-level results to GitHub.
- Keep patient-level outputs and temporary files outside Git tracking.
- Do not commit credentials, tokens, or private information.

## 4. Data and Split Rules

- The formal early-trimester dataset contains 1474 cases.
- Retrospective: 621 cases, existing five-fold development split.
- Multicenter: 442 cases, external test.
- Prospective: 411 cases, prospective test.
- Use the existing formal manifest and fold files.
- Never regenerate or modify the original splits without explicit approval.
- Do not use external test cohorts for model selection or hyperparameter tuning.
- The current grouping key is case_id, not independently verified patient identity.
- Do not automatically delete duplicate-ID candidates.
- Do not treat manifest_all.csv as a replacement for manifest.csv.

## 5. Ground Truth and Measurement Rules

- Follow docs/project/02_GT_PROTOCOL.md and the original reading specification.
- TAR provides two manual measurement lines.
- Gold Mask is used for geometric semantic assignment.
- Do not map TAR endpoints into segmentation-image coordinates.
- Compare the two image sets in the angle domain only.
- Fix manual Card/Reference assignment once per case.
- Never reselect manual GT separately for different model candidates.
- label_gold is manual segmentation ground truth.
- label_infer is a historical predicted mask, not manual ground truth.
- Do not reuse previous experimental measurement algorithms unless
  explicitly approved by the user.

## 6. Research and Implementation Discipline

- Keep the first implementation simple and reproducible.
- Prefer nnUNetv2 for the initial segmentation baseline.
- Do not start model training unless explicitly requested.
- Do not introduce complex transfer-learning methods prematurely.
- Separate segmentation metrics from geometric and clinical measurement metrics.
- Full Angle MAE is the primary clinical measurement metric.
- Always report valid sample counts and invalid-case reasons.

## 7. Engineering Workflow

- Start with read-only inspection and an implementation plan.
- Create code only inside the project repository unless otherwise approved.
- Keep reusable code modular and tested.
- Do not silently change established protocols.
- Do not overwrite existing scripts or configurations.
- Record assumptions, unresolved questions, and deviations.
- Ask the user before destructive operations, data changes,
  split changes, GT-definition changes, or model training.

## 8. Git Discipline

- Repository root: /data5/xjx/zaoyun/data/project1.
- Never add the original data directory to Git.
- Check git status before and after changes.
- Do not push to GitHub without user approval.
- Keep patient-level files and model outputs out of commits.
