# Fire Extinguisher Accessibility

Computer vision research project for automatic assessment of fire-extinguisher accessibility from a single RGB image.

## Research topic

**Автоматическая оценка доступности огнетушителей по изображениям на основе совместного анализа визуального перекрытия и зоны подхода**

Core idea:

> **Visibility is not the same as accessibility.**

A fire extinguisher may remain clearly visible while its approach, grip, or extraction space is blocked.

The project therefore models two complementary quantities:

- **q — visual occlusion:** how much of the extinguisher is directly occluded;
- **r — access obstruction:** how much of the functional manipulation and approach area is occupied.

The final research comparison is:

```text
M1 = q only
M2 = r only
M3 = q + r
```

---

## Dataset

### Base dataset

Initial Fire Extinguisher dataset:

```text
Initial images:              670
After near-duplicate filter: 662
Selected / annotated base:   176
```

Base split:

```text
train: 123
val:    26
test:   27
```

### Synthetic data

53 manually curated synthetic/composite images were used for controlled occlusion and obstruction experiments.

Because several images contain multiple extinguishers, these correspond to:

```text
53 images
61 extinguisher instances
```

### Real extension v3

A final real-image extension contains 28 images:

```text
S0: 10
S1: 10
S2:  3
S3:  5
```

Split:

```text
train: 14
val:    6
test:   8
```

### Final source dataset

```text
train: 196 images
val:    36 images
test:   39 images
total: 271 images
```

Obstacle-positive images:

```text
train: 70
val:    9
test:   6
```

Offline photometric augmentation produces:

```text
train: 1102
val:     36
test:    39
```

Geometry-changing augmentation is avoided to preserve spatial relations.

---

## Segmentation model

The final segmentation baseline uses pretrained **YOLO11n-seg**.

Training configuration:

```text
imgsz:         640
batch:         16
epochs:        50
seed:          42
deterministic: true
GPU:           Tesla T4
```

### Final v3 test results

39 test images, 57 total instances:

| Class             | Mask Precision | Mask Recall | mAP50 | mAP50-95 |
| ----------------- | -------------: | ----------: | ----: | -------: |
| fire_extinguisher |          0.944 |       0.750 | 0.821 |    0.573 |
| obstacle          |          0.000 |       0.000 | 0.000 |    0.000 |

The extinguisher class generalizes reasonably well, while direct prediction of a semantic `obstacle` class does not.

This supports a central methodological conclusion:

> **Obstacle is primarily a spatial/functional relation, not a stable visual category.**

---

## Visual occlusion q

For synthetic images with a known full extinguisher mask:

```text
q = |M_full ∩ M_obstacle| / |M_full|
```

Component analysis:

| Fire mask       | Obstacle mask          |        MAE |       RMSE | Correlation |
| --------------- | ---------------------- | ---------: | ---------: | ----------: |
| GT              | custom YOLO prediction |     0.0741 |     0.1515 |      0.3946 |
| YOLO prediction | GT                     | **0.0124** | **0.0341** |  **0.9749** |
| YOLO prediction | custom YOLO prediction |     0.0858 |     0.1616 |      0.3199 |

The fire mask is accurate enough for the q metric when obstacle ground truth is available. Obstacle perception remains the main bottleneck.

---

## Access obstruction r

`r` is computed from two functional access regions:

- **Z1** — manipulation zone around the extinguisher;
- **Z2** — directional approach corridor from the extinguisher toward the bottom-center of the image.

```text
r1 = |M_obstacle ∩ Z1| / |Z1|
r2 = |M_obstacle ∩ Z2| / |Z2|

r = 0.7 r1 + 0.3 r2
```

If Z2 is not sufficiently visible:

```text
r = r1
```

Final Z2 parameters:

```text
anchor:          (w/2, h-1)
max length:      1.50 * fire_height
width:           max(1.70 * fire_width, 0.45 * fire_height)
minimum length:  0.30 * fire_height
validity:        visibility >= 0.50
```

### Oracle r evaluation

Using GT fire and GT obstacle masks:

| Scenario | Mean r | Median r |
| -------- | -----: | -------: |
| S0       |  0.000 |    0.000 |
| S1       |  0.000 |    0.000 |
| S2       |  0.280 |    0.234 |
| S3       |  0.379 |    0.402 |

This validates the spatial formulation of r independently of perception errors.

---

## Automatic obstacle baselines

Several alternatives were tested:

- direct custom YOLO obstacle class;
- FastSAM generic segmentation;
- pretrained COCO YOLO segmentation;
- YOLOE open-vocabulary segmentation.

YOLOE was the strongest automatic r baseline on the full 28-image v3 set:

```text
MAE:         0.0742
RMSE:        0.1483
Correlation: 0.5604
```

However, after confidence tuning on train+validation, held-out diagnostic performance degraded:

```text
MAE:          0.1734
RMSE:         0.2414
Correlation: -0.0954
```

Therefore obstacle perception is treated as a current limitation rather than used as the final source of q/r.

---

## Accessibility ground truth

Accessibility was manually annotated independently of q and r:

```text
0 = accessible
1 = partially_blocked
2 = severely_blocked
```

Distribution:

```text
class 0: 20
class 1: 30
class 2: 11
total:   61
```

---

## M1 / M2 / M3 evaluation

All models use the same balanced Logistic Regression classifier.

Evaluation uses **5-fold StratifiedGroupKFold**, grouped by source image.

| Model | Features |   Accuracy | Balanced Accuracy |   Macro-F1 |
| ----- | -------- | ---------: | ----------------: | ---------: |
| M1    | q        |     0.4262 |            0.5066 |     0.4185 |
| M2    | r        |     0.6230 |            0.6591 |     0.6165 |
| M3    | q + r    | **0.6557** |        **0.6621** | **0.6439** |

### Repeated grouped cross-validation

50 repetitions of 5-fold StratifiedGroupKFold:

| Model  | Accuracy mean ± std | Macro-F1 mean ± std |
| ------ | ------------------: | ------------------: |
| M1 q   |     0.4193 ± 0.0215 |     0.4177 ± 0.0150 |
| M2 r   |     0.6338 ± 0.0211 |     0.6255 ± 0.0204 |
| M3 q+r | **0.6656 ± 0.0141** | **0.6518 ± 0.0133** |

M3 outperformed M2 in approximately **80%** of repetitions for Accuracy and Macro-F1.

The improvement is modest and uncertainty ranges still include zero, so the hypothesis is interpreted as **partially supported**, not statistically conclusive.

---

## Repository structure

```text
.
├── annotations/          # tracked CVAT / YOLO annotations
├── configs/              # dataset configuration files
├── docs/                 # methodology / project documentation
├── metadata/             # source and split manifests
├── results/
│   ├── final/            # final q/r/accessibility experiment results
│   ├── baselines/        # perception baseline results
│   └── diagnostics/      # audits and diagnostic tables
├── scripts/              # dataset, q/r, evaluation and visualization scripts
├── dataset_guideline.md
├── requirements.txt
└── README.md
```

Large datasets, model weights, generated outputs, training runs and local environments are intentionally excluded from Git.

---

## Main reproducibility scripts

### Dataset

```text
scripts/data_prep/build_source_dataset.py
scripts/data_prep/augment_train_dataset.py
scripts/validate_yolo_dataset.py
```

### q

```text
scripts/q/compute_occlusion_q.py
scripts/q/compute_occlusion_q_instances.py
scripts/q/evaluate_q_synthetic.py
scripts/q/evaluate_q_components.py
```

### r

```text
scripts/r/compute_access_obstruction.py
scripts/r/evaluate_r_gt.py
scripts/r/evaluate_r_coco.py
scripts/r/evaluate_r_yoloe.py
scripts/r/evaluate_r_yoloe_test.py
```

### Final experiment

```text
scripts/r/evaluate_r_synthetic_gt.py
scripts/accessibility/prepare_accessibility_gt.py
scripts/accessibility/evaluate_m1_m2_m3.py
scripts/accessibility/bootstrap_m2_m3.py
scripts/repeated_cv_m1_m2_m3.py
scripts/error_analysis_m2_m3.py
```

---

## Main conclusion

The experiments indicate that:

1. **q alone is insufficient** for accessibility assessment.
2. **r is the strongest individual signal**.
3. Combining **q + r** produces a modest but relatively consistent improvement over r alone.
4. The main unresolved technical bottleneck is robust automatic perception of functionally relevant obstacles from a single RGB image.
