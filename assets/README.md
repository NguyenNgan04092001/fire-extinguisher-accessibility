# Assets

This directory contains small object/obstacle assets used or collected during synthetic-data experiments. It is separate from the final image dataset under `data/`.

## `obstacles/`

Obstacle cutouts extracted during COCO-based experiments. `obstacles_metadata.csv` records COCO image IDs, annotation IDs, source filenames, and mask statistics.

Before redistributing these files publicly, verify the applicable COCO image licenses/attribution requirements for the underlying source images.

## `objects/`

Older/reference object assets collected during early manual-composite experiments. They are **not part of the active reproducible dataset pipeline** and are currently retained only for project history/reference.

Their provenance is not fully documented in the repository. Do not include them in a public dataset/release until source and license information has been verified.

## Active pipeline

The current source-dataset builder does not depend on `assets/objects/`. The final manually composited images are kept locally under `data/` and their final segmentation annotations are stored under `annotations/`.
