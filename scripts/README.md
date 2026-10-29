# Dataset scripts

This directory contains both the active dataset pipeline and historical utilities created while the dataset was being developed.

## Active pipeline

Use these scripts for the current project state:

1. `create_metadata_base.py` — creates the original fixed base split metadata.
2. `split_base_yolo.py` — builds the base YOLO train/val/test dataset from the selected base images and CVAT labels.
3. `prepare_cvat_synthetic.py` — prepares manually composited S1/S2/S3 images for CVAT by copying the corresponding extinguisher labels.
4. `build_source_dataset.py` — **canonical source builder**. Rebuilds `data/fire_accessibility_source` from:
   - base YOLO data;
   - annotated synthetic data;
   - 14 additional real-world images, using `metadata/additional_real_split.csv`.
5. `validate_yolo_dataset.py` — checks image/label parity, class IDs, polygon coordinates, and class counts.
6. `augment_train_dataset.py` — creates photometric augmentations for train only; val/test are copied unchanged.
7. `visualize_yolo_labels.py` — visual sanity check for segmentation annotations.

### Recommended commands

```bash
python scripts/build_source_dataset.py
python scripts/validate_yolo_dataset.py data/fire_accessibility_source
python scripts/augment_train_dataset.py
python scripts/validate_yolo_dataset.py data/fire_accessibility_augmented
```

Only after both validation commands succeed should model training begin.

## Important behavior

`build_source_dataset.py` deletes and rebuilds `data/fire_accessibility_source`. It is safe for the current workflow because the additional real-image split is now stored in the tracked manifest:

```text
metadata/additional_real_split.csv
```

The raw image files themselves remain local under `data/` and are intentionally excluded from Git.

`augment_train_dataset.py` recognizes any synthetic source name ending in `_S<number>` (for example S1, S2, S3, or a future S4), rather than hard-coding only S1–S3.

## One-off / repair utilities

These are not part of the normal training pipeline but are kept for reproducibility and repair:

- `rename_base_images.py`
- `rename_obstacles.py`
- `extract_coco_obstacles.py`
- `convert_disguised_psd.py`
- `fix_synthetic_orientation.py`
- `make_cvat_yolo_zip.py`
- `select_diverse_base.py`

`convert_disguised_psd.py` requires `psd-tools`, which is included in `requirements.txt`.

## Legacy experimental scripts

The following belong to an earlier automatic S1-generation experiment and are **not** the recommended current workflow:

- `generate_s1_candidates.py`
- `generate_s1_occlusion.py`

The final S1–S3 synthetic source images were manually curated/composited and then annotated in CVAT.

## Current high-level pipeline

```text
selected base images + CVAT labels
              ↓
          base_yolo
              ↓
manual S1–S3 composites + CVAT obstacle labels
              ↓
14 additional real images + CVAT labels
              ↓
     build_source_dataset.py
              ↓
     validate_yolo_dataset.py
              ↓
     augment_train_dataset.py
              ↓
     validate_yolo_dataset.py
              ↓
           training
```
