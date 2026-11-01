# Repository audit before model training

Audit performed before starting the first segmentation training experiment.

## Status

**Do not merge/train yet until the annotation-semantic issue below is resolved.**

## Critical methodological issue — modal vs amodal extinguisher masks

The project needs one consistent definition of the `fire_extinguisher` segmentation target.

Two possible definitions exist:

- **modal mask** — only pixels of the extinguisher that are actually visible in the image;
- **amodal/full mask** — the estimated full extinguisher shape, including parts hidden behind an obstacle.

During synthetic-data creation, the original extinguisher polygon was copied from the unobstructed base image before the inserted obstacle was annotated. This can make the synthetic extinguisher target effectively amodal/full. In the additional real images, annotation was performed from the visible image and is naturally modal unless hidden shape was explicitly estimated.

Mixing these definitions in the same training class would make the target inconsistent and would weaken interpretation of segmentation metrics and the later occlusion feature `q`.

### Decision required before training

Choose one policy and apply it consistently:

1. **Recommended for standard YOLO segmentation: modal training masks.**
   - `fire_extinguisher` = visible pixels only.
   - keep a separate full/reference mask for synthetic images if needed to calculate ground-truth occlusion.
   - synthetic visible masks can be derived from the stored full extinguisher mask and obstacle mask.

2. **Amodal segmentation as an explicit research task.**
   - all occluded real images would also need estimated full masks;
   - evaluation must explicitly describe the task as amodal segmentation;
   - this is substantially harder and less aligned with a standard YOLO segmentation baseline.

Until this is decided, the current dataset should be treated as **prepared but not final for training**.

## Evaluation-set limitation

The current augmented split contains only:

```text
val obstacle-positive images:  2
test obstacle-positive images: 2
```

This is sufficient for pipeline smoke testing but not for a scientific comparison of obstacle segmentation or final accessibility performance.

Before paper/defense experiments, expand independent real obstacle cases in validation and especially test.

## Reproducibility issues found and fixed in cleanup branch

### Source dataset rebuild

The old `build_train_with_synthetic.py` rebuilt the source dataset from base + synthetic only. Re-running it after the latest work would have deleted the 14 additional real images from `data/fire_accessibility_source`.

Fixed by replacing it with:

```text
scripts/build_source_dataset.py
```

The additional real split is now tracked in:

```text
metadata/additional_real_split.csv
```

### Future synthetic scenarios

The old augmentation code recognized only S1/S2/S3 names. A future `_S4` image would have been treated as a regular/base image and receive the wrong augmentation count.

Fixed: synthetic recognition now accepts any name ending in `_S<number>`.

### Dataset validation

Added:

```text
scripts/validate_yolo_dataset.py
```

It checks image/label parity, class IDs, polygon coordinate ranges, class instance counts, and obstacle-positive image counts.

### Duplicate export archive

`annotations/cvat_real_additional.zip` duplicated the already extracted tracked annotation directory and was removed.

### Dependencies

`convert_disguised_psd.py` imports `psd_tools`, but `psd-tools` was absent from `requirements.txt`.

Fixed in the cleanup branch.

## Base split reproducibility

`create_metadata_base.py` currently creates the original base split using a fixed random seed, and `split_base_yolo.py` consumes the generated CSV. However the existing base metadata CSV lives under ignored `data/`, so the exact historical split manifest is not currently tracked in Git.

This is not blocking the existing local dataset, but for full reproducibility the exact base split manifest should eventually be copied into a tracked `metadata/` file.

Do **not** regenerate the historical split casually after experiments have started.

## Annotation and dataset files

The following tracked annotation sources are useful and should be retained:

- `annotations/cvat_base/`
- `annotations/cvat_synthetic/`
- `annotations/cvat_real_additional/`

Generated image datasets under `data/` should remain outside Git.

## Assets

`assets/obstacles/` contains extracted COCO obstacle assets with COCO image/annotation IDs recorded in `obstacles_metadata.csv`.

`assets/objects/` appears to contain older object-reference assets and is not referenced by the active dataset pipeline. It is safe to leave it in the private repository for now, but it should be reviewed for source/license provenance before the repository or dataset is made public.

## `src/`

`src/` currently contains only an empty `__init__.py`. This is acceptable as a placeholder. The later accessibility logic (`q`, functional access/extraction zone, `r`, classification) should go here rather than being added to dataset-preparation scripts.

## Recommended pre-training sequence after annotation policy is resolved

```bash
python scripts/build_source_dataset.py
python scripts/validate_yolo_dataset.py data/fire_accessibility_source
python scripts/augment_train_dataset.py
python scripts/validate_yolo_dataset.py data/fire_accessibility_augmented
```

Then verify expected counts before training.

Current expected dataset size, assuming the same source files:

```text
source:
  train 182
  val    30
  test   31

augmented:
  train 1046
  val     30
  test    31
```

## Cleanup assessment

The repository structure is now suitable for continued development. The remaining blocker is primarily **methodological/annotation consistency**, not repository organization.
