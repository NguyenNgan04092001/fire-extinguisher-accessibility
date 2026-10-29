# Metadata

This directory stores small, tracked manifests needed to reproduce the dataset structure without committing the image dataset itself.

## Files

### `additional_real_split.csv`

Fixed train/validation/test assignment for the 14 additional real-world images.

Do not reshuffle these images after model experiments begin.

### `additional_real_sources.csv`

Provenance checklist for the 14 web/real images. Before any public dataset release or paper supplement, fill in for every image:

- original source URL;
- license / usage terms;
- author or attribution when required.

Rows marked `TODO` must not be treated as publication-ready provenance.

## Missing historical base split manifest

The current local project previously generated the base split in:

```text
data/metadata_base.csv
```

Because the whole `data/` directory is intentionally ignored by Git, the exact historical base split CSV is not yet tracked here.

For full reproducibility, copy the **existing local file** (do not regenerate it) into:

```text
metadata/base_split.csv
```

Suggested local command:

```bash
cp data/metadata_base.csv metadata/base_split.csv
```

Then verify that it still reports the established split:

```text
train: 123
val:    26
test:   27
```

Only after checking it should the file be committed. This preserves the exact historical split used to build `data/base_yolo`.
