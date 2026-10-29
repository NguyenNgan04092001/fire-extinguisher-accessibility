# Fire Extinguisher Accessibility

Computer vision project for automatic assessment of fire-extinguisher accessibility from RGB images.

## Research topic

**Автоматическая оценка доступности огнетушителей по изображениям на основе совместного анализа визуального перекрытия и зоны подхода**

The core idea is that **visibility is not the same as accessibility**. A fire extinguisher may be clearly visible but still difficult to retrieve because of objects or structures that block the approach, grip, or extraction space.

The functional access zone may include:

- the area in front of the extinguisher;
- lateral / hand-access space;
- space around the handle;
- upper extraction space required to lift or remove the extinguisher;
- cabinet/holder constraints only when they actually restrict extraction.

A cabinet itself is **not automatically an obstacle**.

## Current segmentation classes

```text
0 fire_extinguisher
1 obstacle
```

## Current dataset status

Source dataset:

```text
train: 182 images / 182 labels
val:    30 images / 30 labels
test:   31 images / 31 labels
```

Augmented dataset:

```text
train: 1046 images / 1046 labels
val:     30 images / 30 labels
test:    31 images / 31 labels
```

Obstacle-positive images in the augmented dataset:

```text
train: 534
val:     2
test:    2
```

The 1046 training images are **not 1046 independent scenes**. They are generated from 182 source training images using controlled photometric augmentation.

## Project structure

```text
.
├── annotations/        # tracked CVAT / YOLO segmentation annotations
├── assets/             # small obstacle/reference assets
├── configs/            # YOLO dataset YAML files
├── docs/               # project status and repository/methodology audit
├── metadata/           # tracked split manifests
├── results/            # lightweight annotation checks / visual outputs
├── scripts/            # dataset preparation and validation utilities
├── src/                # reusable accessibility-analysis code (future)
├── dataset_guideline.md
├── requirements.txt
└── README.md
```

Large image datasets, generated augmented datasets, model weights, and training runs are intentionally excluded from Git.

## Reproducible dataset pipeline

```text
base dataset + annotations
        ↓
synthetic annotations
        ↓
additional real images + fixed split manifest
        ↓
build_source_dataset.py
        ↓
validate_yolo_dataset.py
        ↓
augment_train_dataset.py
        ↓
validate_yolo_dataset.py
        ↓
model training
```

Recommended commands:

```bash
python scripts/build_source_dataset.py
python scripts/validate_yolo_dataset.py data/fire_accessibility_source
python scripts/augment_train_dataset.py
python scripts/validate_yolo_dataset.py data/fire_accessibility_augmented
```

The additional real-image split is stored in:

```text
metadata/additional_real_split.csv
```

## Important pre-training blocker

Training is intentionally paused while one annotation decision is resolved: the `fire_extinguisher` target must use a consistent **modal (visible-only)** or **amodal/full** mask definition.

Synthetic composites may currently inherit a full extinguisher mask from the unobstructed base image, while real-image annotations are naturally visible-only. These semantics should not be mixed in a final training dataset.

See:

- [`docs/repository_audit.md`](docs/repository_audit.md)
- [`docs/project_status.md`](docs/project_status.md)
- [`dataset_guideline.md`](dataset_guideline.md)

## Evaluation caveat

Validation and test currently contain only two obstacle-positive images each. This is sufficient for pipeline verification but should be expanded before final scientific comparisons or publication claims.

## Script documentation

See [`scripts/README.md`](scripts/README.md) for the active pipeline, repair utilities, and legacy experimental scripts.
