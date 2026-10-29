from pathlib import Path
import random

import cv2
import numpy as np


BASE_ROOT = Path("data/base_yolo")
MANUAL_ROOT = Path("data/manual_composite")
OUT_ROOT = Path("data/s1_candidates")

OBJECT_DIRS = [
    Path("assets/objects_nobg"),
    Path("assets/obstacles"),
]

SPLITS = ["train", "val", "test"]

CANDIDATES_PER_IMAGE = 3
MAX_TRIES = 100

# S1: gần như không được che bình
MAX_Q = 0.02

# S1: không được chắn mạnh vùng phía trước bình
MAX_INTRUSION = 0.15

random.seed(42)


def get_objects():
    objects = []

    for folder in OBJECT_DIRS:
        if not folder.exists():
            continue

        for path in folder.rglob("*.png"):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                objects.append(path)

    return objects


def manual_exists(split, stem):
    folder = MANUAL_ROOT / split / "S1"

    for ext in [".png", ".jpg", ".jpeg"]:
        if (folder / f"{stem}_S1{ext}").exists():
            return True

    return False


def read_masks(label_path, w, h):
    masks = []

    with open(label_path) as f:
        for line in f:
            values = line.strip().split()

            if len(values) < 7:
                continue

            coords = list(map(float, values[1:]))

            points = []

            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)

                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)

            cv2.fillPoly(mask, [points], 1)

            masks.append(mask)

    return masks


def get_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        xs.min(),
        ys.min(),
        xs.max(),
        ys.max(),
    )


def resize_object(obj, extinguisher_height, image_h, image_w):
    h, w = obj.shape[:2]

    new_h = int(
        extinguisher_height *
        random.uniform(0.55, 1.15)
    )

    # Không cho object quá cao
    new_h = min(
        new_h,
        int(image_h * 0.35)
    )

    new_h = max(new_h, 30)

    new_w = int(
        w * new_h / h
    )

    # Không cho object quá rộng
    if new_w > image_w * 0.40:

        scale = (
            image_w * 0.40 /
            new_w
        )

        new_w = int(new_w * scale)
        new_h = int(new_h * scale)

    return cv2.resize(
        obj,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def paste_object(base, obj, x, y):
    h, w = base.shape[:2]
    oh, ow = obj.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(w, x + ow)
    y2 = min(h, y + oh)

    if x1 >= x2 or y1 >= y2:
        return None

    sx1 = x1 - x
    sy1 = y1 - y

    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    crop = obj[
        sy1:sy2,
        sx1:sx2
    ]

    rgb = np.zeros_like(base)

    alpha = np.zeros(
        (h, w),
        dtype=np.float32
    )

    rgb[y1:y2, x1:x2] = crop[:, :, :3]

    alpha[
        y1:y2,
        x1:x2
    ] = (
        crop[:, :, 3] /
        255.0
    )

    a = alpha[:, :, None]

    result = (
        rgb * a +
        base * (1 - a)
    ).astype(np.uint8)

    mask = (
        alpha > 0.5
    ).astype(np.uint8)

    return result, mask


def generate_candidate(
    base,
    extinguisher_mask,
    bbox,
    objects
):

    h, w = base.shape[:2]

    x1, y1, x2, y2 = bbox

    fire_h = y2 - y1

    # vùng tiếp cận heuristic
    fire_w = x2 - x1

    corridor_left = max(
        0,
        int(x1 - fire_w * 0.5)
    )

    corridor_right = min(
        w,
        int(x2 + fire_w * 0.5)
    )

    corridor = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    corridor[
        y2:h,
        corridor_left:corridor_right
    ] = 1

    fire_area = max(
        1,
        extinguisher_mask.sum()
    )

    for _ in range(MAX_TRIES):

        obj_path = random.choice(objects)

        obj = cv2.imread(
            str(obj_path),
            cv2.IMREAD_UNCHANGED
        )

        obj = resize_object(
            obj,
            fire_h,
            h,
            w
        )

        oh, ow = obj.shape[:2]

        side = random.choice(
            ["left", "right"]
        )

        gap = int(
            random.uniform(0.02, 0.08) * w
        )

        if side == "left":

            x = (
                x1 -
                gap -
                ow
            )

        else:

            x = (
                x2 +
                gap
            )

        # QUAN TRỌNG:
        # Neo object xuống đáy ảnh
        bottom = int(
            h *
            random.uniform(
                0.97,
                1.03
            )
        )

        y = bottom - oh

        placed = paste_object(
            base,
            obj,
            x,
            y
        )

        if placed is None:
            continue

        result, obj_mask = placed

        obj_area = obj_mask.sum()

        if obj_area < 50:
            continue

        # q: mức che bình
        overlap = (
            obj_mask *
            extinguisher_mask
        ).sum()

        q = overlap / fire_area

        if q > MAX_Q:
            continue

        # mức chắn corridor
        intrusion = (
            obj_mask *
            corridor
        ).sum() / obj_area

        if intrusion > MAX_INTRUSION:
            continue

        return (
            result,
            obj_path.name,
            q,
            intrusion
        )

    return None


def main():

    objects = get_objects()

    print(
        "Objects:",
        len(objects)
    )

    for split in SPLITS:

        image_dir = (
            BASE_ROOT /
            "images" /
            split
        )

        label_dir = (
            BASE_ROOT /
            "labels" /
            split
        )

        output_dir = (
            OUT_ROOT /
            split
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in sorted(
            image_dir.glob("*")
        ):

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue

            stem = image_path.stem

            # bỏ qua 20 ảnh đã làm tay
            if manual_exists(
                split,
                stem
            ):
                print(
                    "[manual skip]",
                    stem
                )
                continue

            label_path = (
                label_dir /
                f"{stem}.txt"
            )

            base = cv2.imread(
                str(image_path)
            )

            h, w = base.shape[:2]

            masks = read_masks(
                label_path,
                w,
                h
            )

            if not masks:
                continue

            # bình lớn nhất
            fire_mask = max(
                masks,
                key=lambda x: x.sum()
            )

            bbox = get_bbox(
                fire_mask
            )

            for i in range(
                1,
                CANDIDATES_PER_IMAGE + 1
            ):

                candidate = generate_candidate(
                    base,
                    fire_mask,
                    bbox,
                    objects
                )

                if candidate is None:

                    print(
                        "[FAIL]",
                        stem,
                        i
                    )

                    continue

                image, obj_name, q, intrusion = candidate

                output_name = (
                    f"{stem}_S1_c{i}.png"
                )

                cv2.imwrite(
                    str(
                        output_dir /
                        output_name
                    ),
                    image
                )

                print(
                    output_name,
                    obj_name,
                    f"q={q:.3f}",
                    f"r≈{intrusion:.3f}"
                )


if __name__ == "__main__":
    main()