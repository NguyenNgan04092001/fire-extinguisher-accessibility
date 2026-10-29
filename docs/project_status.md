# Project status

## Current research direction

The project evaluates fire-extinguisher accessibility from a single RGB image.

The current formulation extends beyond simple front-area obstruction. Accessibility should reflect whether a person can approach, grip, lift, and extract the extinguisher. Therefore, the relevant functional access zone may include:

- the area in front of the extinguisher;
- lateral / side space required for hand access;
- free space around the handle;
- upper extraction space required to lift or remove the extinguisher;
- structural constraints from cabinets or holders when they actually restrict extraction.

A cabinet itself is not automatically an obstacle.

## Current segmentation labels

```text
0 fire_extinguisher
1 obstacle
```

## Current source dataset

```text
train: 182 images
val:    30 images
test:   31 images
```

The latest 14 additional real-world images use the fixed split stored in:

```text
metadata/additional_real_split.csv
```

## Current augmented dataset

```text
train: 1046 images / 1046 labels
val:     30 images / 30 labels
test:    31 images / 31 labels
```

Obstacle-containing images:

```text
train: 534
val:     2
test:    2
```

Important: augmentation increases the number of training samples but does not create 1046 independent scenes.

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
training
```

The source dataset no longer requires manual copy commands to restore the 14 real images.

## Training status

**Training is intentionally paused.**

A repository audit found an annotation-semantic issue that must be resolved first: the `fire_extinguisher` target may be amodal/full in synthetic composites (because the original base mask was copied before adding the obstacle) while real-image annotations are naturally modal/visible-only.

Mixing modal and amodal targets in one class is not recommended.

See [`repository_audit.md`](repository_audit.md) for the detailed finding and the recommended options.

## Evaluation caveat

The current validation and test sets contain only two obstacle-positive images each. This is enough for pipeline verification, but not sufficient for strong scientific conclusions about obstacle segmentation or accessibility classification.

Before final paper/defense experiments, the evaluation set should be expanded with more independent real-world obstacle cases.

## Next steps

1. Decide and standardize `fire_extinguisher` mask semantics (modal vs amodal).
2. If using modal masks, convert/re-annotate synthetic extinguisher masks while preserving full reference masks separately for occlusion analysis.
3. Run the source and augmented dataset validators.
4. Expand independent validation/test obstacle cases for final scientific evaluation.
5. Train the YOLO segmentation baseline.
6. Evaluate segmentation metrics separately for `fire_extinguisher` and `obstacle`.
7. Implement accessibility logic under `src/` (occlusion, functional access/extraction zone, classification).
8. Compare with alternative modern segmentation models using the same split and metrics.
9. Prepare error analysis and paper-ready experiment tables.
