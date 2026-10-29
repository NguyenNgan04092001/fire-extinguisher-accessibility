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

The latest 14 additional real-world images were split as follows:

```text
train: real_0001, real_0003, real_0005, real_0006, real_0007, real_0010
val:   real_0002, real_0004, real_0008, real_0011
test:  real_0012, real_0013, real_0014, real_0015
```

Among those images, obstacle annotations are present in representative real-world cases such as door / chair / clutter / equipment / upper extraction constraint.

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

## Current pipeline

```text
base dataset
    ↓
CVAT annotation of fire_extinguisher
    ↓
manual synthetic obstruction scenarios
    ↓
CVAT annotation of obstacle
    ↓
merge into source dataset
    ↓
add real-world accessibility cases
    ↓
photometric augmentation of training split only
    ↓
YOLO segmentation training
```

## Training status

Training is currently paused intentionally.

The next model experiment was planned with YOLO11n-seg, but repository cleanup and methodology clarification are being completed first.

## Methodological caveat

The current validation and test sets contain only two obstacle-positive images each. This is enough for pipeline verification, but not sufficient for strong scientific conclusions about obstacle segmentation or accessibility classification.

Before the final paper / defense, the evaluation set should be expanded with more independent real-world obstacle cases.

## Next steps

1. Finalize the operational definition of accessibility.
2. Add / refine annotation rules for front, side, handle, and extraction-space constraints.
3. Expand independent validation and test obstacle cases.
4. Train the YOLO segmentation baseline.
5. Evaluate segmentation metrics separately for `fire_extinguisher` and `obstacle`.
6. Implement accessibility features / rules based on segmentation outputs.
7. Compare the proposed approach with alternative modern segmentation models under the same data split and metrics.
8. Prepare an error analysis and scientific-paper-ready experiment table.
