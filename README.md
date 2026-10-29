# Fire Extinguisher Accessibility

Computer vision project for automatic assessment of fire-extinguisher accessibility from RGB images.

## Research topic

**Автоматическая оценка доступности огнетушителей по изображениям на основе совместного анализа визуального перекрытия и зоны подхода**

The core idea is that **visibility is not the same as accessibility**. A fire extinguisher may be clearly visible but still difficult to retrieve because of objects or structures that block the approach, grip, or extraction space.

The current project therefore considers two complementary aspects:

- visual occlusion of the fire extinguisher;
- obstruction of the functional access / extraction zone around the extinguisher.

The access zone is not limited to the floor area directly in front of the extinguisher. It can also include lateral space and the space above / around the handle that is required to grip, lift, or remove the extinguisher.

## Current segmentation classes

```text
0 fire_extinguisher
1 obstacle
```

A cabinet or fixed structure is **not automatically labeled as an obstacle**. It is considered an obstacle only when it clearly restricts access or extraction.

## Current dataset status

Source dataset after the latest update:

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

Obstacle-containing images in the augmented dataset:

```text
train: 534
val:     2
test:    2
```

The 1046 training images are **not 1046 independent scenes**. They are generated from 182 source training images using controlled photometric augmentation.

## Dataset composition

The current source dataset combines:

- base fire-extinguisher images;
- manually composited synthetic obstruction scenarios;
- additional real-world images containing cabinets, nearby structures, front/side obstructions, and extraction-space constraints.

The additional real images were deliberately split across train, validation, and test so that obstacle cases are present outside the training set.

## Project structure

```text
.
├── annotations/        # CVAT / YOLO segmentation annotations
├── assets/             # small project assets and visual material
├── configs/            # YOLO dataset YAML files
├── docs/               # research notes and current project status
├── results/            # lightweight checks / visual outputs
├── scripts/            # dataset preparation utilities
├── src/                # reusable project code
├── dataset_guideline.md
├── requirements.txt
└── README.md
```

Large image datasets, generated augmented datasets, model weights, and training runs are intentionally excluded from Git.

## Dataset pipeline

```text
raw / selected images
        ↓
CVAT segmentation
        ↓
source dataset
        ↓
controlled photometric augmentation
        ↓
training dataset
        ↓
segmentation model
        ↓
accessibility analysis
```

Core dataset scripts are documented in [`scripts/README.md`](scripts/README.md).

## Main dataset configs

- `configs/base_yolo.yaml`
- `configs/fire_accessibility_source.yaml`
- `configs/fire_accessibility_augmented.yaml`

## Important annotation rule

For the accessibility task, the project distinguishes between:

- an object being **near** the extinguisher;
- an object actually **blocking** approach, grip, or extraction.

This distinction is important for hard-negative cases and for avoiding the incorrect rule `near object = inaccessible`.

## Current state

Dataset preparation is complete enough for the next experimental stage, but model training is intentionally paused while the repository and methodology are being cleaned and documented.

See [`docs/project_status.md`](docs/project_status.md) for the current state and next steps.
