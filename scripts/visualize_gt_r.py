from pathlib import Path
import cv2
import numpy as np

from compute_access_obstruction import create_access_zones
from evaluate_r_gt import load_gt_mask


ROOT = Path("data/real_extension_v3/selected")
ANN = Path("annotations/cvat_real_extension_v3")
OUT = Path("outputs/gt_r_debug")

CASES = [
    ("S2", "val", "v3_0022.png"),
    ("S2", "test", "v3_0023.png"),
    ("S3", "test", "v3_0028.png"),
]


def overlay(img, mask, color, alpha):
    m = mask.astype(bool)
    img[m] = (
        (1 - alpha) * img[m]
        + alpha * np.array(color)
    ).astype(np.uint8)


OUT.mkdir(parents=True, exist_ok=True)

for scenario, split, name in CASES:
    img_path = ROOT / scenario / name
    label_path = ANN / split / "labels" / f"{Path(name).stem}.txt"

    img = cv2.imread(str(img_path))

    fire = load_gt_mask(
        label_path,
        img.shape,
        class_id=0,
    )

    obstacle = load_gt_mask(
        label_path,
        img.shape,
        class_id=1,
    )

    z1, z2, info = create_access_zones(fire)

    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # yellow = Z2
    overlay(vis, z2, [255, 255, 0], 0.30)

    # red = Z1
    overlay(vis, z1, [255, 0, 0], 0.35)

    # blue = GT obstacle
    overlay(vis, obstacle, [0, 0, 255], 0.65)

    # green = fire
    overlay(vis, fire, [0, 255, 0], 0.75)

    z1_overlap = int((obstacle & z1).sum())
    z2_overlap = int((obstacle & z2).sum())

    print(
        name,
        "GT obstacle area =", int(obstacle.sum()),
        "Z1 overlap =", z1_overlap,
        "Z2 overlap =", z2_overlap,
    )

    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        str(OUT / f"{Path(name).stem}_gt_debug.jpg"),
        vis,
    )

print("Saved to:", OUT)