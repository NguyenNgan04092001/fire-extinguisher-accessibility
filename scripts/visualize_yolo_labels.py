from pathlib import Path
import random

import cv2
import numpy as np


IMAGE_DIR = Path("data/base_yolo/images/train")
LABEL_DIR = Path("data/base_yolo/labels/train")
OUTPUT_DIR = Path("results/annotation_check")

NUM_IMAGES = 10
RANDOM_SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    random.seed(RANDOM_SEED)
    selected = random.sample(
        image_files,
        min(NUM_IMAGES, len(image_files))
    )

    for image_path in selected:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Không đọc được ảnh: {image_path}")
            continue

        h, w = image.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Không có label: {label_path}")
            continue

        with label_path.open("r") as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()

            if len(values) < 7:
                continue

            class_id = int(values[0])

            coords = list(map(float, values[1:]))

            points = []

            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)

                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            # Vẽ polygon
            cv2.polylines(
                image,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            # Tô nhẹ vùng polygon
            overlay = image.copy()

            cv2.fillPoly(
                overlay,
                [points],
                color=(0, 255, 0)
            )

            image = cv2.addWeighted(
                overlay,
                0.25,
                image,
                0.75,
                0
            )

            # Ghi class id
            x0, y0 = points[0]

            cv2.putText(
                image,
                f"fire_extinguisher ({class_id})",
                (x0, max(20, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            image
        )

        print(f"Đã lưu: {output_path}")

    print("\nHoàn thành.")
    print(f"Kiểm tra ảnh tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()