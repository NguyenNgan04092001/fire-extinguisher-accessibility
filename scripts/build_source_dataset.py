#!/usr/bin/env python3
"""Build the current source dataset reproducibly.

Inputs (stored locally unless noted):
- data/base_yolo/{images,labels}/{train,val,test}
- data/cvat_synthetic/images
- annotations/cvat_synthetic/labels/train          [tracked]
- data/additional_real/raw
- annotations/cvat_real_additional/labels/train    [tracked]
- metadata/additional_real_split.csv               [tracked]

Output:
- data/fire_accessibility_source
- configs/fire_accessibility_source.yaml

The output directory is rebuilt from scratch. Unlike the older
build_train_with_synthetic.py, this script also restores the additional
real-world images in their fixed train/val/test splits.
"""

from pathlib import Path
import csv
import shutil
import sys

BASE_ROOT = Path("data/base_yolo")
BASE_AUDIT_LABELS = Path("annotations/cvat_potential_obstacle_audit/labels/train")
SYN_IMAGES = Path("data/cvat_synthetic/images")
SYN_LABELS = Path("annotations/cvat_synthetic/labels/train")
REAL_IMAGES = Path("data/additional_real/raw")
REAL_LABELS = Path("annotations/cvat_real_additional/labels/train")
REAL_SPLIT_CSV = Path("metadata/additional_real_split.csv")

OUT_ROOT = Path("data/fire_accessibility_source")
YAML_PATH = Path("configs/fire_accessibility_source.yaml")

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_CLASSES = {0, 1}


def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def label_classes(label_path: Path):
    classes = set()
    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid class id in {label_path}:{line_no}: {parts[0]}"
            ) from exc
        classes.add(cls)
    return classes


def validate_label(label_path: Path, require_extinguisher=True):
    classes = label_classes(label_path)
    unknown = classes - VALID_CLASSES
    if unknown:
        raise ValueError(f"Unknown class(es) {sorted(unknown)} in {label_path}")
    if require_extinguisher and 0 not in classes:
        raise ValueError(f"Missing class 0 fire_extinguisher in {label_path}")
    return classes


def ensure_output_dirs():
    for split in SPLITS:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(img_path: Path, label_path: Path, split: str, source: str):
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}' for {img_path.name}")
    if not img_path.exists():
        raise FileNotFoundError(f"Missing {source} image: {img_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing {source} label: {label_path}")

    validate_label(label_path)

    dst_img = OUT_ROOT / "images" / split / img_path.name
    dst_lbl = OUT_ROOT / "labels" / split / f"{img_path.stem}.txt"

    if dst_img.exists() or dst_lbl.exists():
        raise FileExistsError(
            f"Duplicate output stem '{img_path.stem}' while adding {source}"
        )

    shutil.copy2(img_path, dst_img)
    shutil.copy2(label_path, dst_lbl)


def copy_base_dataset():
    counts = {}
    for split in SPLITS:
        img_dir = BASE_ROOT / "images" / split
        lbl_dir = BASE_ROOT / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(
                f"Base split is missing: {img_dir} or {lbl_dir}"
            )

        count = 0
        for img_path in image_files(img_dir):
            base_label_path = lbl_dir / f"{img_path.stem}.txt"
            audit_label_path = BASE_AUDIT_LABELS / f"{img_path.stem}.txt"

            if audit_label_path.exists():
                label_path = audit_label_path
                source = "base potential-obstacle audit"
            else:
                label_path = base_label_path
                source = "base"

            copy_pair(img_path, label_path, split, source)
            count += 1
        counts[split] = count
    return counts


def copy_synthetic_train():
    if not SYN_IMAGES.exists():
        raise FileNotFoundError(f"Missing synthetic images: {SYN_IMAGES}")
    if not SYN_LABELS.exists():
        raise FileNotFoundError(f"Missing synthetic labels: {SYN_LABELS}")

    count = 0
    obstacle_count = 0
    for img_path in image_files(SYN_IMAGES):
        label_path = SYN_LABELS / f"{img_path.stem}.txt"
        classes = validate_label(label_path)
        if 1 not in classes:
            raise ValueError(
                f"Synthetic image is expected to contain obstacle class 1: "
                f"{label_path}"
            )
        copy_pair(img_path, label_path, "train", "synthetic")
        count += 1
        obstacle_count += 1
    return count, obstacle_count


def copy_additional_real():
    if not REAL_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Missing real split manifest: {REAL_SPLIT_CSV}")
    if not REAL_IMAGES.exists():
        raise FileNotFoundError(
            f"Missing local real images: {REAL_IMAGES}. "
            "Restore real_0001...real_0015 (except real_0009) before rebuilding."
        )
    if not REAL_LABELS.exists():
        raise FileNotFoundError(f"Missing real labels: {REAL_LABELS}")

    counts = {split: 0 for split in SPLITS}
    obstacle_counts = {split: 0 for split in SPLITS}

    with REAL_SPLIT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"file_name", "split", "has_obstacle"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest must contain columns {sorted(required)}: {REAL_SPLIT_CSV}"
            )

        seen = set()
        for row in reader:
            file_name = row["file_name"].strip()
            split = row["split"].strip()
            expected_obstacle = row["has_obstacle"].strip().lower() == "yes"

            if file_name in seen:
                raise ValueError(f"Duplicate real image in manifest: {file_name}")
            seen.add(file_name)

            img_path = REAL_IMAGES / file_name
            label_path = REAL_LABELS / f"{Path(file_name).stem}.txt"
            classes = validate_label(label_path)
            actual_obstacle = 1 in classes

            if actual_obstacle != expected_obstacle:
                raise ValueError(
                    f"Obstacle flag mismatch for {file_name}: manifest="
                    f"{expected_obstacle}, label={actual_obstacle}"
                )

            copy_pair(img_path, label_path, split, "additional real")
            counts[split] += 1
            if actual_obstacle:
                obstacle_counts[split] += 1

    return counts, obstacle_counts


def validate_output():
    summary = {}
    for split in SPLITS:
        img_dir = OUT_ROOT / "images" / split
        lbl_dir = OUT_ROOT / "labels" / split

        images = image_files(img_dir)
        image_stems = {p.stem for p in images}
        label_paths = sorted(lbl_dir.glob("*.txt"))
        label_stems = {p.stem for p in label_paths}

        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels or missing_images:
            raise RuntimeError(
                f"Image/label mismatch in {split}: "
                f"missing_labels={missing_labels[:5]}, "
                f"missing_images={missing_images[:5]}"
            )

        obstacle_images = 0
        for label_path in label_paths:
            classes = validate_label(label_path)
            if 1 in classes:
                obstacle_images += 1

        summary[split] = {
            "images": len(images),
            "labels": len(label_paths),
            "obstacle_images": obstacle_images,
        }
    return summary


def write_yaml():
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(
        f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: fire_extinguisher
  1: obstacle
""",
        encoding="utf-8",
    )


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    ensure_output_dirs()

    print("Copy base dataset...")
    base_counts = copy_base_dataset()
    print(base_counts)

    print("\nAdd synthetic obstacle images to train...")
    syn_count, syn_obstacle_count = copy_synthetic_train()
    print(f"synthetic: {syn_count}, with obstacle: {syn_obstacle_count}")

    print("\nAdd additional real images using fixed manifest...")
    real_counts, real_obstacle_counts = copy_additional_real()
    print(f"real: {real_counts}")
    print(f"real obstacle images: {real_obstacle_counts}")

    write_yaml()
    summary = validate_output()

    print("\n" + "=" * 60)
    print("SOURCE DATASET READY")
    print("=" * 60)
    for split in SPLITS:
        s = summary[split]
        print(
            f"{split:5s}: {s['images']} images, {s['labels']} labels, "
            f"{s['obstacle_images']} obstacle-positive"
        )
    print(f"Dataset: {OUT_ROOT}")
    print(f"YAML   : {YAML_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        sys.exit(1)
