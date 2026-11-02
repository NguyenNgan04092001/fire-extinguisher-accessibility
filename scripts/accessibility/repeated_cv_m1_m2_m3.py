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
)


QR_CSV = Path(
    "outputs/qr_synthetic_instances.csv"
)

GT_CSV = Path(
    "outputs/accessibility_gt/"
    "accessibility_gt_template.csv"
)

OUT = Path(
    "outputs/repeated_cv_m1_m2_m3.csv"
)

N_SPLITS = 5
N_REPEATS = 50


def make_model(seed):
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
                random_state=seed,
            ),
        ),
    ])


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def run_one_repeat(
    X,
    y,
    groups,
    seed,
):
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    predictions = np.full(
        len(y),
        -1,
        dtype=int,
    )

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups,
    ):
        model = make_model(seed)

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        predictions[test_idx] = (
            model.predict(
                X[test_idx]
            )
        )

    if np.any(predictions < 0):
        raise RuntimeError(
            "Some samples were not predicted."
        )

    return compute_metrics(
        y,
        predictions,
    )


def summarize(values):
    arr = np.asarray(
        values,
        dtype=float,
    )

    low, high = np.percentile(
        arr,
        [2.5, 97.5],
    )

    return {
        "mean": arr.mean(),
        "std": arr.std(ddof=1),
        "median": np.median(arr),
        "low": low,
        "high": high,
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

    y = df[
        "accessibility_gt"
    ].to_numpy()

    groups = df[
        "image"
    ].to_numpy()

    models = {
        "M1_q":
            df[["q"]].to_numpy(
                dtype=float
            ),

        "M2_r":
            df[["r"]].to_numpy(
                dtype=float
            ),

        "M3_q_r":
            df[["q", "r"]].to_numpy(
                dtype=float
            ),
    }

    results = []

    print(
        f"Instances : {len(df)}"
    )

    print(
        f"Images    : "
        f"{df['image'].nunique()}"
    )

    print(
        f"Repeats   : {N_REPEATS}"
    )

    print(
        f"Folds     : {N_SPLITS}"
    )

    print()

    for repeat in range(
        N_REPEATS
    ):
        seed = 1000 + repeat

        row = {
            "repeat": repeat + 1,
            "seed": seed,
        }

        for model_name, X in models.items():

            metrics = run_one_repeat(
                X,
                y,
                groups,
                seed,
            )

            for metric_name, value in (
                metrics.items()
            ):
                row[
                    f"{model_name}_"
                    f"{metric_name}"
                ] = value

        # Deltas M3 - M2
        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]:
            row[
                f"delta_M3_M2_"
                f"{metric_name}"
            ] = (
                row[
                    f"M3_q_r_"
                    f"{metric_name}"
                ]
                -
                row[
                    f"M2_r_"
                    f"{metric_name}"
                ]
            )

        results.append(row)

        print(
            f"[{repeat + 1:02d}/"
            f"{N_REPEATS}] "
            f"M1 F1="
            f"{row['M1_q_macro_f1']:.3f} "
            f"M2 F1="
            f"{row['M2_r_macro_f1']:.3f} "
            f"M3 F1="
            f"{row['M3_q_r_macro_f1']:.3f} "
            f"delta="
            f"{row['delta_M3_M2_macro_f1']:+.3f}"
        )

    result_df = pd.DataFrame(
        results
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUT,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "REPEATED STRATIFIED GROUP CV SUMMARY"
    )
    print("=" * 78)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:

        print()
        print(
            metric.upper()
        )

        for model_name in [
            "M1_q",
            "M2_r",
            "M3_q_r",
        ]:
            values = result_df[
                f"{model_name}_{metric}"
            ].to_numpy()

            s = summarize(values)

            print(
                f"{model_name:8s} "
                f"mean={s['mean']:.4f} "
                f"std={s['std']:.4f} "
                f"median={s['median']:.4f} "
                f"95% range="
                f"[{s['low']:.4f}, "
                f"{s['high']:.4f}]"
            )

    print()
    print("=" * 78)
    print(
        "M3 - M2 DELTA ACROSS REPEATS"
    )
    print("=" * 78)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        values = result_df[
            f"delta_M3_M2_{metric}"
        ].to_numpy()

        s = summarize(values)

        positive = (
            values > 0
        ).mean()

        equal_or_better = (
            values >= 0
        ).mean()

        print()
        print(metric)

        print(
            f"  mean delta       = "
            f"{s['mean']:+.4f}"
        )

        print(
            f"  median delta     = "
            f"{s['median']:+.4f}"
        )

        print(
            f"  std              = "
            f"{s['std']:.4f}"
        )

        print(
            f"  95% range        = "
            f"[{s['low']:+.4f}, "
            f"{s['high']:+.4f}]"
        )

        print(
            f"  P(M3 > M2)       = "
            f"{positive:.3f}"
        )

        print(
            f"  P(M3 >= M2)      = "
            f"{equal_or_better:.3f}"
        )

    print()
    print(
        f"Saved: {OUT}"
    )


if __name__ == "__main__":
    main()