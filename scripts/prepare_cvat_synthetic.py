#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import shutil
import sys

import cv2

MANUAL_ROOT = Path("data/manual_composite")
BASE_ROOT = Path("data/base_yolo")
OUT_ROOT = Path("data/cvat_synthetic")

SPLITS = ["train", "val", "test"]
SCENARIOS = ["S1", "S2", "S3"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CLEAN_OUTPUT = True

BASE_ID_RE = re.compile(r"^(base_\d+)", re.IGNORECASE)


def get_image_size(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        return None

    h, w = img.shape[:2]
    return int(w), int(h)


def extract_base_id(filename: str):
    stem = Path(filename).stem
    m = BASE_ID_RE.match(stem)
    return m.group(1) if m else None


def find_base_image(base_id: str, expected_split: str):
    search_order = [expected_split] + [s for s in SPLITS if s != expected_split]
    for split in search_order:
        image_dir = BASE_ROOT / "images" / split
        if not image_dir.exists():
            continue
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = image_dir / f"{base_id}{ext}"
            if p.exists():
                return p, split
    return None, None


def find_base_label(base_id: str, base_split: str):
    if base_split is None:
        return None
    p = BASE_ROOT / "labels" / base_split / f"{base_id}.txt"
    return p if p.exists() else None


def normalize_output_name(src_path: Path, base_id: str, scenario: str):
    return f"{base_id}_{scenario}{src_path.suffix.lower()}"


def ensure_dirs():
    if CLEAN_OUTPUT and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for sub in [
        "images",
        "labels",
        "rejected/size_mismatch",
        "rejected/missing_base",
        "rejected/missing_label",
        "rejected/unreadable",
        "rejected/duplicate",
    ]:
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()

    manifest = []
    mismatches = []
    missing_labels = []
    seen_output_stems = set()

    stats = {
        "source_images": 0,
        "ok": 0,
        "size_mismatch": 0,
        "missing_base": 0,
        "missing_label": 0,
        "unreadable": 0,
        "duplicate": 0,
        "psd_ignored": 0,
        "wrong_split": 0,
    }

    for split in SPLITS:
        for scenario in SCENARIOS:
            src_dir = MANUAL_ROOT / split / scenario
            if not src_dir.exists():
                continue

            print(f"\n===== {split.upper()} / {scenario} =====")

            for src_path in sorted(p for p in src_dir.iterdir() if p.is_file()):
                if src_path.suffix.lower() == ".psd":
                    stats["psd_ignored"] += 1
                    print(f"[IGNORE PSD] {src_path.name}")
                    continue

                if src_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                stats["source_images"] += 1
                base_id = extract_base_id(src_path.name)

                row = {
                    "status": "",
                    "expected_split": split,
                    "actual_base_split": "",
                    "scenario": scenario,
                    "base_id": base_id or "",
                    "source_file": str(src_path),
                    "output_image": "",
                    "output_label": "",
                    "base_image": "",
                    "base_label": "",
                    "synthetic_size": "",
                    "base_size": "",
                    "note": "",
                }

                if base_id is None:
                    row["status"] = "missing_base_id"
                    row["note"] = "Không parse được base_xxxx từ tên file."
                    manifest.append(row)
                    stats["missing_base"] += 1
                    print(f"[FAIL NAME] {src_path.name}")
                    continue

                out_name = normalize_output_name(src_path, base_id, scenario)
                out_stem = Path(out_name).stem

                if out_stem in seen_output_stems:
                    row["status"] = "duplicate"
                    row["note"] = f"Trùng synthetic stem: {out_stem}"
                    reject_path = OUT_ROOT / "rejected" / "duplicate" / out_name
                    shutil.copy2(src_path, reject_path)
                    row["output_image"] = str(reject_path)
                    manifest.append(row)
                    stats["duplicate"] += 1
                    print(f"[DUPLICATE] {src_path.name} -> {out_stem}")
                    continue

                seen_output_stems.add(out_stem)

                base_image, actual_base_split = find_base_image(base_id, split)

                if base_image is None:
                    row["status"] = "missing_base_image"
                    row["note"] = f"Không tìm thấy ảnh base cho {base_id}"
                    reject_path = OUT_ROOT / "rejected" / "missing_base" / out_name
                    shutil.copy2(src_path, reject_path)
                    row["output_image"] = str(reject_path)
                    manifest.append(row)
                    stats["missing_base"] += 1
                    print(f"[MISSING BASE] {src_path.name}")
                    continue

                row["base_image"] = str(base_image)
                row["actual_base_split"] = actual_base_split

                if actual_base_split != split:
                    stats["wrong_split"] += 1
                    row["note"] = (
                        f"WARNING: synthetic ở {split} nhưng base thuộc {actual_base_split}."
                    )

                base_label = find_base_label(base_id, actual_base_split)

                if base_label is None:
                    row["status"] = "missing_base_label"
                    reject_path = OUT_ROOT / "rejected" / "missing_label" / out_name
                    shutil.copy2(src_path, reject_path)
                    row["output_image"] = str(reject_path)

                    missing_labels.append({
                        "split": split,
                        "scenario": scenario,
                        "base_id": base_id,
                        "synthetic_file": src_path.name,
                        "expected_label": str(
                            BASE_ROOT / "labels" / actual_base_split / f"{base_id}.txt"
                        ),
                    })

                    manifest.append(row)
                    stats["missing_label"] += 1
                    print(f"[MISSING LABEL] {src_path.name}")
                    continue

                row["base_label"] = str(base_label)

                syn_size = get_image_size(src_path)
                base_size = get_image_size(base_image)

                if syn_size is None:
                    row["status"] = "unreadable_synthetic"
                    row["note"] = "OpenCV không đọc được ảnh synthetic."
                    reject_path = OUT_ROOT / "rejected" / "unreadable" / out_name
                    shutil.copy2(src_path, reject_path)
                    row["output_image"] = str(reject_path)
                    manifest.append(row)
                    stats["unreadable"] += 1
                    print(f"[UNREADABLE] {src_path.name}")
                    continue

                if base_size is None:
                    row["status"] = "unreadable_base"
                    row["note"] = "OpenCV không đọc được ảnh base."
                    reject_path = OUT_ROOT / "rejected" / "unreadable" / out_name
                    shutil.copy2(src_path, reject_path)
                    row["output_image"] = str(reject_path)
                    manifest.append(row)
                    stats["unreadable"] += 1
                    print(f"[UNREADABLE BASE] {base_image}")
                    continue

                row["synthetic_size"] = f"{syn_size[0]}x{syn_size[1]}"
                row["base_size"] = f"{base_size[0]}x{base_size[1]}"

                if syn_size != base_size:
                    row["status"] = "size_mismatch"
                    msg = (
                        f"Kích thước synthetic {syn_size[0]}x{syn_size[1]} "
                        f"khác base {base_size[0]}x{base_size[1]}."
                    )
                    row["note"] = (row["note"] + " " + msg).strip()

                    reject_path = OUT_ROOT / "rejected" / "size_mismatch" / out_name
                    shutil.copy2(src_path, reject_path)
                    row["output_image"] = str(reject_path)

                    mismatches.append({
                        "split": split,
                        "scenario": scenario,
                        "base_id": base_id,
                        "synthetic_file": src_path.name,
                        "synthetic_size": row["synthetic_size"],
                        "base_image": base_image.name,
                        "base_size": row["base_size"],
                    })

                    manifest.append(row)
                    stats["size_mismatch"] += 1
                    print(
                        f"[SIZE MISMATCH] {src_path.name}: "
                        f"{row['synthetic_size']} != {row['base_size']}"
                    )
                    continue

                out_image = OUT_ROOT / "images" / out_name
                out_label = OUT_ROOT / "labels" / f"{out_stem}.txt"

                shutil.copy2(src_path, out_image)
                shutil.copy2(base_label, out_label)

                row["status"] = "ok"
                row["output_image"] = str(out_image)
                row["output_label"] = str(out_label)

                manifest.append(row)
                stats["ok"] += 1

                print(
                    f"[OK] {src_path.name} -> {out_name} | "
                    f"{row['synthetic_size']} | label={out_label.name}"
                )

    manifest_fields = [
        "status",
        "expected_split",
        "actual_base_split",
        "scenario",
        "base_id",
        "source_file",
        "output_image",
        "output_label",
        "base_image",
        "base_label",
        "synthetic_size",
        "base_size",
        "note",
    ]

    write_csv(OUT_ROOT / "manifest.csv", manifest, manifest_fields)

    write_csv(
        OUT_ROOT / "size_mismatches.csv",
        mismatches,
        [
            "split",
            "scenario",
            "base_id",
            "synthetic_file",
            "synthetic_size",
            "base_image",
            "base_size",
        ],
    )

    write_csv(
        OUT_ROOT / "missing_labels.csv",
        missing_labels,
        [
            "split",
            "scenario",
            "base_id",
            "synthetic_file",
            "expected_label",
        ],
    )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)
    print(f"Ảnh render tìm thấy      : {stats['source_images']}")
    print(f"Hợp lệ + đã copy label   : {stats['ok']}")
    print(f"Sai kích thước           : {stats['size_mismatch']}")
    print(f"Thiếu ảnh base           : {stats['missing_base']}")
    print(f"Thiếu label base         : {stats['missing_label']}")
    print(f"Không đọc được ảnh       : {stats['unreadable']}")
    print(f"Duplicate                : {stats['duplicate']}")
    print(f"PSD bỏ qua               : {stats['psd_ignored']}")
    print(f"Synthetic sai split base : {stats['wrong_split']}")
    print()
    print(f"Ảnh OK   : {OUT_ROOT / 'images'}")
    print(f"Labels OK: {OUT_ROOT / 'labels'}")
    print(f"Manifest : {OUT_ROOT / 'manifest.csv'}")

    if stats["size_mismatch"] > 0:
        print(
            "\nCẢNH BÁO: Có ảnh sai kích thước. "
            "KHÔNG import label cũ cho các ảnh đó."
        )

    if stats["wrong_split"] > 0:
        print(
            "\nCẢNH BÁO: Có synthetic nằm khác split với base gốc. "
            "Kiểm tra manifest.csv."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)
