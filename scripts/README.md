# Dataset scripts

The `scripts/` directory contains utilities created during dataset preparation. They do not all belong to the final training pipeline.

## Core pipeline scripts

These scripts are currently the most important for reproducing the active dataset workflow:

- `prepare_cvat_synthetic.py` — prepares manually composited synthetic images and corresponding extinguisher labels for CVAT.
- `build_train_with_synthetic.py` — merges the base YOLO dataset with synthetic obstacle annotations into the source accessibility dataset.
- `augment_train_dataset.py` — creates the augmented training dataset using photometric transformations while leaving validation and test unchanged.
- `split_base_yolo.py` — creates the original base train/val/test split.
- `create_metadata_base.py` — creates metadata for the selected base images.
- `visualize_yolo_labels.py` — visual inspection of YOLO segmentation labels.

## Data-preparation / one-off utilities

These were useful during earlier preparation stages but are not part of the normal end-to-end pipeline:

- `rename_base_images.py`
- `rename_obstacles.py`
- `extract_coco_obstacles.py`
- `convert_disguised_psd.py`
- `fix_synthetic_orientation.py`
- `make_cvat_yolo_zip.py`
- `select_diverse_base.py`

Keep these scripts because they document how the dataset was built and may be useful for reproducibility or repair.

## Legacy / experimental synthetic generation

The following scripts belong to an earlier automatic S1 generation attempt and should not be treated as the current recommended synthetic-data workflow:

- `generate_s1_candidates.py`
- `generate_s1_occlusion.py`

The current synthetic obstruction set was manually curated / composited and then annotated in CVAT.

## Current recommended execution order

For the existing project state, the relevant sequence is approximately:

```text
base selection / split
        ↓
CVAT base annotation
        ↓
manual synthetic composition
        ↓
prepare_cvat_synthetic.py
        ↓
CVAT obstacle annotation
        ↓
build_train_with_synthetic.py
        ↓
add additional real images + labels to source split
        ↓
augment_train_dataset.py
        ↓
model training
```

Large image datasets are stored locally under `data/` and are intentionally excluded from Git.
