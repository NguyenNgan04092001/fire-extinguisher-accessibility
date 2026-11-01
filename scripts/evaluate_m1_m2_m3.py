#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


QR_CSV = Path(
    "outputs/qr_synthetic_instances.csv"
)

GT_CSV = Path(
    "outputs/accessibility_gt/"
    "accessibility_gt_template.csv"
)

OUT_CSV = Path(
    "outputs/m1_m2_m3_predictions.csv"
)

RANDOM_STATE = 42
N_SPLITS = 5


def make_model():
    return Pipeline([
        (
            "scale",
            StandardScaler(),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def evaluate_model(
    name,
    X,
    y,
    groups,
    cv,
):
    predictions = np.full(
        len(y),
        -1,
        dtype=int,
    )

    fold_ids = np.full(
        len(y),
        -1,
        dtype=int,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        model = make_model()

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        pred = model.predict(
            X[test_idx]
        )

        predictions[test_idx] = pred
        fold_ids[test_idx] = fold

    accuracy = accuracy_score(
        y,
        predictions,
    )

    balanced = balanced_accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2],
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"Accuracy          = "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy = "
        f"{balanced:.4f}"
    )

    print(
        f"Macro F1          = "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1       = "
        f"{weighted_f1:.4f}"
    )

    print()
    print(
        "Confusion matrix "
        "(rows=GT, cols=Pred):"
    )

    print(cm)

    print()
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "accessible",
                "partially_blocked",
                "severely_blocked",
            ],
            digits=3,
            zero_division=0,
        )
    )

    return {
        "predictions": predictions,
        "fold_ids": fold_ids,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    qr = pd.read_csv(
        QR_CSV
    )

    gt = pd.read_csv(
        GT_CSV
    )

    gt = gt[
        [
            "image",
            "instance_id",
            "accessibility_gt",
        ]
    ].copy()

    df = qr.merge(
        gt,
        on=[
            "image",
            "instance_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    df["accessibility_gt"] = (
        df["accessibility_gt"]
        .astype(int)
    )

    if len(df) != 61:
        raise RuntimeError(
            f"Expected 61 instances, "
            f"got {len(df)}"
        )

    print(
        f"Merged instances: {len(df)}"
    )

    print()
    print(
        "GT class distribution:"
    )

    print(
        df[
            "accessibility_gt"
        ]
        .value_counts()
        .sort_index()
    )

    y = (
        df["accessibility_gt"]
        .to_numpy()
    )

    # Group by original image.
    # Multiple extinguishers from one image
    # must remain in the same fold.
    groups = (
        df["image"]
        .to_numpy()
    )

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # -------------------------
    # M1 — q only
    # -------------------------

    X1 = (
        df[["q"]]
        .to_numpy(
            dtype=float
        )
    )

    result_m1 = evaluate_model(
        "M1 — q only",
        X1,
        y,
        groups,
        cv,
    )

    # -------------------------
    # M2 — r only
    # -------------------------

    X2 = (
        df[["r"]]
        .to_numpy(
            dtype=float
        )
    )

    # Re-create CV object for clarity
    cv2 = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    result_m2 = evaluate_model(
        "M2 — r only",
        X2,
        y,
        groups,
        cv2,
    )

    # -------------------------
    # M3 — q + r
    # -------------------------

    X3 = (
        df[["q", "r"]]
        .to_numpy(
            dtype=float
        )
    )

    cv3 = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    result_m3 = evaluate_model(
        "M3 — q + r",
        X3,
        y,
        groups,
        cv3,
    )

    # -------------------------
    # SUMMARY
    # -------------------------

    print()
    print("=" * 72)
    print("M1 / M2 / M3 SUMMARY")
    print("=" * 72)

    print(
        f"{'Model':<15}"
        f"{'Accuracy':>12}"
        f"{'Balanced':>12}"
        f"{'Macro-F1':>12}"
    )

    for name, result in [
        ("M1 q", result_m1),
        ("M2 r", result_m2),
        ("M3 q+r", result_m3),
    ]:
        print(
            f"{name:<15}"
            f"{result['accuracy']:>12.4f}"
            f"{result['balanced_accuracy']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
        )

    # -------------------------
    # SAVE
    # -------------------------

    df[
        "pred_M1_q"
    ] = result_m1[
        "predictions"
    ]

    df[
        "pred_M2_r"
    ] = result_m2[
        "predictions"
    ]

    df[
        "pred_M3_qr"
    ] = result_m3[
        "predictions"
    ]

    df[
        "cv_fold"
    ] = result_m3[
        "fold_ids"
    ]

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    print()
    print(
        f"Saved: {OUT_CSV}"
    )


if __name__ == "__main__":
    main()