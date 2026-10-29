from pathlib import Path
import csv
import math
import cv2
import torch
from ultralytics import YOLO

ROOT = Path("data/fire_accessibility_source")
OUT = Path("data/audit/s4_candidates")
OUT.mkdir(parents=True, exist_ok=True)

CSV_OUT = Path("data/audit/s4_candidates_auto.csv")

# COCO objects that can plausibly obstruct access
TARGET_NAMES = {
    "chair",
    "bench",
    "couch",
    "backpack",
    "handbag",
    "suitcase",
    "potted plant",
    "dining table",
    "bed",
    "bicycle",
    "motorcycle",
}

# Generous screening threshold.
# This does NOT decide whether an object is truly an obstruction.
MAX_NORMALIZED_GAP = 1.25
CONF = 0.25


def read_fire_boxes(label_path, img_w, img_h):
    """Read class-0 YOLO segmentation polygons and return pixel bboxes."""
    fire_boxes = []
    has_obstacle = False

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue

        cls = int(float(parts[0]))

        if cls == 1:
            has_obstacle = True

        if cls != 0:
            continue

        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]

        x1 = min(xs) * img_w
        y1 = min(ys) * img_h
        x2 = max(xs) * img_w
        y2 = max(ys) * img_h

        fire_boxes.append((x1, y1, x2, y2))

    return fire_boxes, has_obstacle


def rect_gap(a, b):
    """Minimum Euclidean gap between two rectangles."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    dx = max(ax1 - bx2, bx1 - ax2, 0)
    dy = max(ay1 - by2, by1 - ay2, 0)

    return math.hypot(dx, dy)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# Collect only images that currently have no class-1 annotation
items = []

for split in ["train", "val"]:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        fire_boxes, has_obstacle = read_fire_boxes(label_path, w, h)

        if not fire_boxes:
            continue

        # Existing positive obstacle images are not S4-missing-label candidates.
        if has_obstacle:
            continue

        items.append({
            "split": split,
            "image_path": image_path,
            "fire_boxes": fire_boxes,
        })


device = "cpu"

print("Device:", device)
print("Images to screen:", len(items))

model = YOLO("yolo11n.pt")

item_by_path = {
    str(x["image_path"].resolve()): x
    for x in items
}

rows = []
candidate_images = set()

results = model.predict(
    source=[str(x["image_path"]) for x in items],
    conf=CONF,
    device=device,
    verbose=False,
    stream=True,
)

for result in results:
    image_path = Path(result.path)
    key = str(image_path.resolve())

    item = item_by_path.get(key)
    if item is None:
        continue

    img = cv2.imread(str(image_path))
    if img is None:
        continue

    found_any = False

    # draw extinguisher reference boxes
    for fb in item["fire_boxes"]:
        x1, y1, x2, y2 = map(int, fb)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = model.names[cls_id]

            if name not in TARGET_NAMES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            obj_box = (x1, y1, x2, y2)

            best_gap = float("inf")
            best_norm_gap = float("inf")
            best_iou = 0.0

            for fire_box in item["fire_boxes"]:
                gap = rect_gap(fire_box, obj_box)

                fx1, fy1, fx2, fy2 = fire_box
                diag = math.hypot(fx2 - fx1, fy2 - fy1)
                norm_gap = gap / max(diag, 1.0)

                iou = bbox_iou(fire_box, obj_box)

                if norm_gap < best_norm_gap:
                    best_gap = gap
                    best_norm_gap = norm_gap
                    best_iou = iou

            if best_norm_gap > MAX_NORMALIZED_GAP:
                continue

            found_any = True
            candidate_images.add((item["split"], image_path.name))

            rows.append({
                "split": item["split"],
                "image": image_path.name,
                "object": name,
                "confidence": round(conf, 4),
                "normalized_gap": round(best_norm_gap, 4),
                "bbox_iou_with_fire": round(best_iou, 4),
            })

            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))

            cv2.rectangle(img, p1, p2, (0, 200, 255), 2)

            text = f"{name} {conf:.2f} gap={best_norm_gap:.2f}"
            cv2.putText(
                img,
                text,
                (p1[0], max(20, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

    if found_any:
        out_path = OUT / f"{item['split']}__{image_path.name}"
        cv2.imwrite(str(out_path), img)


with CSV_OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "image",
            "object",
            "confidence",
            "normalized_gap",
            "bbox_iou_with_fire",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


print()
print("===================================")
print("S4 CANDIDATE SCREENING COMPLETE")
print("===================================")
print("Images screened :", len(items))
print("Candidate images:", len(candidate_images))
print("Candidate objects:", len(rows))
print("CSV             :", CSV_OUT)
print("Preview folder  :", OUT)
