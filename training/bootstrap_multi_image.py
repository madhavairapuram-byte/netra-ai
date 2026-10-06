import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "reports/external_validation_v4_predictions.csv"

OUTPUT_FILE = "reports/bootstrap_multi_image_results.csv"

N_SIMULATIONS = 1000

GROUP_SIZES = [1, 3, 5]

THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05
)

RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "true_label",
    "conjunctivitis_probability"
]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}"
        )

# ============================================================
# LABELS
# ============================================================

df["true_binary"] = (
    df["true_label"]
    .str.lower()
    .map({
        "normal": 0,
        "conjunctivitis": 1
    })
)

df["probability"] = pd.to_numeric(
    df["conjunctivitis_probability"],
    errors="coerce"
)

if df["true_binary"].isna().any():
    raise ValueError(
        "Unknown labels detected."
    )

if df["probability"].isna().any():
    raise ValueError(
        "Invalid probability values detected."
    )

# ============================================================
# SEPARATE CLASSES
# ============================================================

normal = df[
    df["true_binary"] == 0
]["probability"].values

conjunctivitis = df[
    df["true_binary"] == 1
]["probability"].values

print("=" * 70)
print("BOOTSTRAP MULTI-IMAGE ANALYSIS")
print("=" * 70)

print(
    f"\nNormal images: {len(normal)}"
)

print(
    f"Conjunctivitis images: {len(conjunctivitis)}"
)

print(
    f"Simulations per group size: {N_SIMULATIONS}"
)

# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

# ============================================================
# STORAGE
# ============================================================

all_results = []

# ============================================================
# RUN SIMULATIONS
# ============================================================

for group_size in GROUP_SIZES:

    print("\n")
    print("=" * 70)
    print(
        f"{group_size}-IMAGE SIMULATION"
    )
    print("=" * 70)

    for simulation in range(
        N_SIMULATIONS
    ):

        # ----------------------------------------------------
        # Number of complete groups
        # ----------------------------------------------------

        n_normal_groups = (
            len(normal) // group_size
        )

        n_conj_groups = (
            len(conjunctivitis)
            // group_size
        )

        n_groups = min(
            n_normal_groups,
            n_conj_groups
        )

        # ----------------------------------------------------
        # Randomly shuffle independently
        # ----------------------------------------------------

        normal_shuffled = rng.permutation(
            normal
        )

        conj_shuffled = rng.permutation(
            conjunctivitis
        )

        # ----------------------------------------------------
        # Create groups
        # ----------------------------------------------------

        normal_group_probabilities = []

        for i in range(n_groups):

            start = i * group_size
            end = start + group_size

            group = normal_shuffled[
                start:end
            ]

            mean_probability = np.mean(
                group
            )

            normal_group_probabilities.append(
                mean_probability
            )

        conj_group_probabilities = []

        for i in range(n_groups):

            start = i * group_size
            end = start + group_size

            group = conj_shuffled[
                start:end
            ]

            mean_probability = np.mean(
                group
            )

            conj_group_probabilities.append(
                mean_probability
            )

        y_true = np.array(
            [0] * n_groups +
            [1] * n_groups
        )

        y_probability = np.array(
            normal_group_probabilities +
            conj_group_probabilities
        )

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        auc = roc_auc_score(
            y_true,
            y_probability
        )

        # ----------------------------------------------------
        # Test thresholds
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            y_pred = (
                y_probability >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            sensitivity = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            # Specificity
            true_negative = np.sum(
                (y_true == 0) &
                (y_pred == 0)
            )

            false_positive = np.sum(
                (y_true == 0) &
                (y_pred == 1)
            )

            specificity = (
                true_negative /
                (true_negative + false_positive)
                if (
                    true_negative +
                    false_positive
                ) > 0
                else 0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            all_results.append({
                "group_size": group_size,
                "simulation": simulation + 1,
                "threshold": round(
                    float(threshold),
                    2
                ),
                "ROC_AUC": auc,
                "accuracy": accuracy,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1
            })

# ============================================================
# DATAFRAME
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY FUNCTION
# ============================================================

def confidence_interval(values):

    values = np.array(
        values
    )

    mean = np.mean(
        values
    )

    lower = np.percentile(
        values,
        2.5
    )

    upper = np.percentile(
        values,
        97.5
    )

    return mean, lower, upper


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BOOTSTRAP SUMMARY")
print("=" * 70)

for group_size in GROUP_SIZES:

    group = results[
        results["group_size"] ==
        group_size
    ]

    # --------------------------------------------------------
    # Select best threshold based on average F1
    # --------------------------------------------------------

    threshold_summary = (
        group
        .groupby("threshold")
        .agg({
            "f1": "mean"
        })
        .reset_index()
    )

    best_threshold = threshold_summary.loc[
        threshold_summary["f1"].idxmax(),
        "threshold"
    ]

    selected = group[
        group["threshold"] ==
        best_threshold
    ]

    print("\n")
    print(
        f"{group_size} IMAGE(S)"
    )

    print(
        f"Selected threshold: "
        f"{best_threshold:.2f}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = [
        "ROC_AUC",
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1"
    ]

    for metric in metrics:

        mean, lower, upper = (
            confidence_interval(
                selected[metric]
            )
        )

        print(
            f"{metric:<15} "
            f"{mean:.3f} "
            f"(95% CI "
            f"{lower:.3f}–{upper:.3f})"
        )

# ============================================================
# COMPARE 1 VS 3 VS 5
# ============================================================

print("\n")
print("=" * 70)
print("COMPARISON")
print("=" * 70)

for group_size in GROUP_SIZES:

    group = results[
        results["group_size"] ==
        group_size
    ]

    threshold_summary = (
        group
        .groupby("threshold")
        .agg({
            "f1": "mean"
        })
        .reset_index()
    )

    best_threshold = threshold_summary.loc[
        threshold_summary["f1"].idxmax(),
        "threshold"
    ]

    selected = group[
        group["threshold"] ==
        best_threshold
    ]

    accuracy_mean = selected[
        "accuracy"
    ].mean()

    auc_mean = selected[
        "ROC_AUC"
    ].mean()

    sensitivity_mean = selected[
        "sensitivity"
    ].mean()

    specificity_mean = selected[
        "specificity"
    ].mean()

    f1_mean = selected[
        "f1"
    ].mean()

    print(
        f"\n{group_size} image(s)"
    )

    print(
        f"  Threshold:   {best_threshold:.2f}"
    )

    print(
        f"  Accuracy:    {accuracy_mean:.3f}"
    )

    print(
        f"  ROC-AUC:     {auc_mean:.3f}"
    )

    print(
        f"  Sensitivity: {sensitivity_mean:.3f}"
    )

    print(
        f"  Specificity: {specificity_mean:.3f}"
    )

    print(
        f"  F1:          {f1_mean:.3f}"
    )

# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nFull simulation results saved to:"
)

print(
    OUTPUT_FILE
)