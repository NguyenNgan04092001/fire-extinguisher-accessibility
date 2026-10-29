from pathlib import Path
import csv
import random

import numpy as np
import requests
from PIL import Image
from pycocotools.coco import COCO


# =========================
# CẤU HÌNH
# =========================

ANNOTATION_FILE = Path(
    "data/coco/annotations/instances_train2017.json"
)

CACHE_DIR = Path("data/coco/downloaded_images")

OUTPUT_ROOT = Path("assets/obstacles")

CATEGORIES = {
    "chair": 30,
    "suitcase": 30,
    "bench": 30,
    "potted plant": 30,
}

RANDOM_SEED = 42

# Loại object quá nhỏ
MIN_WIDTH = 80
MIN_HEIGHT = 80
MIN_AREA = 4000

# Mask phải chiếm tối thiểu bao nhiêu phần bbox
MIN_MASK_RATIO = 0.25


def download_image(file_name: str) -> Path:
    """
    Chỉ tải ảnh COCO khi thực sự cần.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CACHE_DIR / file_name

    if output_path.exists():
        return output_path

    url = (
        "http://images.cocodataset.org/"
        f"train2017/{file_name}"
    )

    print(f"Tải: {url}")

    response = requests.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)

    return output_path


def extract_rgba(
    image: Image.Image,
    mask: np.ndarray,
    bbox
):
    """
    Crop object theo bbox và dùng mask làm alpha channel.
    """

    x, y, w, h = bbox

    x1 = max(0, int(x))
    y1 = max(0, int(y))

    x2 = min(
        image.width,
        int(x + w)
    )

    y2 = min(
        image.height,
        int(y + h)
    )

    image_np = np.array(
        image.convert("RGB")
    )

    crop_rgb = image_np[
        y1:y2,
        x1:x2
    ]

    crop_mask = mask[
        y1:y2,
        x1:x2
    ]

    alpha = (
        crop_mask.astype(np.uint8) * 255
    )

    rgba = np.dstack(
        [crop_rgb, alpha]
    )

    return Image.fromarray(
        rgba,
        mode="RGBA"
    )


def main():

    random.seed(RANDOM_SEED)

    print("Đọc COCO annotations...")

    coco = COCO(
        str(ANNOTATION_FILE)
    )

    metadata_rows = []

    for category_name, target_count in CATEGORIES.items():

        print(
            f"\n===== {category_name.upper()} ====="
        )

        output_dir = (
            OUTPUT_ROOT / category_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        category_ids = coco.getCatIds(
            catNms=[category_name]
        )

        if not category_ids:
            print(
                f"Không tìm thấy class: {category_name}"
            )
            continue

        ann_ids = coco.getAnnIds(
            catIds=category_ids,
            iscrowd=False
        )

        annotations = coco.loadAnns(
            ann_ids
        )

        random.shuffle(annotations)

        saved = 0

        for ann in annotations:

            if saved >= target_count:
                break

            x, y, w, h = ann["bbox"]

            # -------------------------
            # Lọc object quá nhỏ
            # -------------------------

            if w < MIN_WIDTH:
                continue

            if h < MIN_HEIGHT:
                continue

            if ann["area"] < MIN_AREA:
                continue

            bbox_area = w * h

            if bbox_area <= 0:
                continue

            mask_ratio = (
                ann["area"] / bbox_area
            )

            if mask_ratio < MIN_MASK_RATIO:
                continue

            # -------------------------
            # Lấy thông tin ảnh
            # -------------------------

            image_info = coco.loadImgs(
                [ann["image_id"]]
            )[0]

            file_name = image_info[
                "file_name"
            ]

            try:
                image_path = download_image(
                    file_name
                )

                image = Image.open(
                    image_path
                ).convert("RGB")

            except Exception as e:

                print(
                    f"[LỖI TẢI ẢNH] {file_name}: {e}"
                )

                continue

            # -------------------------
            # Tạo binary mask
            # -------------------------

            mask = coco.annToMask(
                ann
            )

            if mask.sum() == 0:
                continue

            # -------------------------
            # Crop RGBA
            # -------------------------

            rgba = extract_rgba(
                image,
                mask,
                ann["bbox"]
            )

            saved += 1

            output_name = (
                f"{category_name}_{saved:03d}.png"
            )

            output_path = (
                output_dir / output_name
            )

            rgba.save(
                output_path
            )

            print(
                f"[{saved}/{target_count}] "
                f"{output_path}"
            )

            metadata_rows.append({
                "category": category_name,
                "output_file": str(output_path),
                "coco_image_id": ann["image_id"],
                "coco_annotation_id": ann["id"],
                "source_file": file_name,
                "bbox_width": round(w, 2),
                "bbox_height": round(h, 2),
                "mask_area": round(
                    ann["area"],
                    2
                ),
                "mask_bbox_ratio": round(
                    mask_ratio,
                    4
                ),
            })

        print(
            f"Đã lưu {saved} "
            f"{category_name}"
        )

    # =========================
    # LƯU METADATA
    # =========================

    metadata_path = (
        OUTPUT_ROOT /
        "obstacles_metadata.csv"
    )

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "category",
            "output_file",
            "coco_image_id",
            "coco_annotation_id",
            "source_file",
            "bbox_width",
            "bbox_height",
            "mask_area",
            "mask_bbox_ratio",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            metadata_rows
        )

    print("\n========== HOÀN THÀNH ==========")

    for category_name in CATEGORIES:
        count = len(
            list(
                (
                    OUTPUT_ROOT /
                    category_name
                ).glob("*.png")
            )
        )

        print(
            f"{category_name}: {count}"
        )

    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()