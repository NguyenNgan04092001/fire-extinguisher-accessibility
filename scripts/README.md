# Scripts

Scripts used to build the dataset and reproduce the q/r accessibility experiments.

## Dataset pipeline

- `create_metadata_base.py`
- `split_base_yolo.py`
- `prepare_cvat_synthetic.py`
- `build_source_dataset.py`
- `validate_yolo_dataset.py`
- `augment_train_dataset.py`

## q — visual occlusion

- `compute_occlusion_q.py` — single-image q calculation.
- `compute_occlusion_q_instances.py` — instance-level q calculation.
- `evaluate_q_synthetic.py` — q statistics on the synthetic dataset.
- `audit_q_consistency.py` — annotation consistency audit.
- `evaluate_q_yolo.py` — visible-mask baseline.
- `evaluate_q_components.py` — GT/predicted mask component analysis.
- `evaluate_q_yoloe.py` — YOLOE obstacle baseline.

## r — access obstruction

- `compute_access_obstruction.py` — creates Z1/Z2 and computes r.
- `evaluate_r_gt.py` — oracle evaluation on `real_extension_v3`.
- `evaluate_r_v3.py` — FastSAM baseline.
- `evaluate_r_coco.py` — closed-set COCO baseline.
- `evaluate_r_yoloe.py` — YOLOE open-vocabulary baseline.
- `tune_r_yoloe.py` — confidence tuning using train + validation only.
- `evaluate_r_yoloe_test.py` — held-out diagnostic evaluation.
- `evaluate_r_synthetic_gt.py` — computes q+r features for the synthetic extinguisher instances.

## Accessibility experiment

- `prepare_accessibility_gt.py` — prepares images and CSV for independent accessibility labeling.
- `label_accessibility_gt.py` — labeling utility.
- `evaluate_m1_m2_m3.py` — grouped 5-fold comparison:
  - M1 = q
  - M2 = r
  - M3 = q+r
- `bootstrap_m2_m3.py` — cluster bootstrap comparison.
- `repeated_cv_m1_m2_m3.py` — 50× repeated stratified grouped CV.
- `error_analysis_m2_m3.py` — compares errors corrected and introduced by M3.

## Visualization / diagnostics

- `visualize_gt_r.py`
- `visualize_yoloe_debug.py`
- `test_fire_inference.py`

## Historical / repair utilities

The repository also keeps several one-off scripts used during dataset construction, including image renaming, PSD conversion, obstacle extraction, candidate generation and CVAT preparation.

These files are not part of the final experimental pipeline but are retained for reproducibility.
