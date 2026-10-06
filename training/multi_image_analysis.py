import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "reports/external_validation_v4_predictions.csv"

OUTPUT_FILE = "reports/multi_image_simulation_results.csv"

# Number of images combined into one assessment
GROUP_SIZES = [1, 3, 5]

# Candidate thresholds
THRESHOLDS = np.arange(0.50, 0.96, 0.05)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "filename",
    "true_label",
    "conjunctivitis_probability"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

# ============================================================
# CONVERT LABELS
# ============================================================

df["true_binary"] = (
    df["true_label"]
    .str.lower()
    .map({
        "normal": 0,
        "conjunctivitis": 1
    })
)

if df["true_binary"].isna().any():
    raise ValueError(
        "Unknown labels found in true_label."
    )

df["probability"] = pd.to_numeric(
    df["conjunctivitis_probability"],
    errors="coerce"
)

if df["probability"].isna().any():
    raise ValueError(
        "Invalid probability values found."
    )

# ============================================================
# FUNCTION: CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_prob, threshold):

    y_pred = (
        np.array(y_prob) >= threshold
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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# SINGLE IMAGE BASELINE
# ============================================================

print("=" * 70)
print("MULTI-IMAGE CONJUNCTIVITIS SCREENING ANALYSIS")
print("=" * 70)

print("\nImages available:", len(df))

print(
    "Normal:",
    (df["true_binary"] == 0).sum()
)

print(
    "Conjunctivitis:",
    (df["true_binary"] == 1).sum()
)

# ============================================================
# IMPORTANT NOTE
# ============================================================

print("\nNOTE:")
print(
    "The current external dataset does not contain verified "
    "longitudinal images from the same eye."
)

print(
    "Therefore this script performs a controlled simulation "
    "of multi-image probability aggregation."
)

print(
    "It must NOT be interpreted as evidence from real "
    "multi-timepoint patient imaging."
)

# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)

all_results = []

# ============================================================
# GROUP IMAGES
# ============================================================

for group_size in GROUP_SIZES:

    print("\n")
    print("=" * 70)
    print(
        f"GROUP SIZE: {group_size} IMAGE(S)"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Create balanced groups
    # --------------------------------------------------------

    normal = df[
        df["true_binary"] == 0
    ].copy()

    conjunctivitis = df[
        df["true_binary"] == 1
    ].copy()

    rng.shuffle(normal.values)
    rng.shuffle(conjunctivitis.values)

    # Number of complete groups
    normal_groups = len(normal) // group_size
    conjunctivitis_groups = (
        len(conjunctivitis) // group_size
    )

    # Keep equal number of groups
    number_of_groups = min(
        normal_groups,
        conjunctivitis_groups
    )

    print(
        "Normal groups:",
        number_of_groups
    )

    print(
        "Conjunctivitis groups:",
        number_of_groups
    )

    grouped_true = []
    grouped_prob = []

    # --------------------------------------------------------
    # Normal groups
    # --------------------------------------------------------

    for i in range(number_of_groups):

        start = i * group_size
        end = start + group_size

        group = normal.iloc[
            start:end
        ]

        probabilities = (
            group["probability"].values
        )

        # Mean probability
        combined_probability = np.mean(
            probabilities
        )

        grouped_true.append(0)
        grouped_prob.append(
            combined_probability
        )

    # --------------------------------------------------------
    # Conjunctivitis groups
    # --------------------------------------------------------

    for i in range(number_of_groups):

        start = i * group_size
        end = start + group_size

        group = conjunctivitis.iloc[
            start:end
        ]

        probabilities = (
            group["probability"].values
        )

        combined_probability = np.mean(
            probabilities
        )

        grouped_true.append(1)
        grouped_prob.append(
            combined_probability
        )

    grouped_true = np.array(
        grouped_true
    )

    grouped_prob = np.array(
        grouped_prob
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        grouped_true,
        grouped_prob
    )

    print(
        f"\nSimulated ROC-AUC: {auc:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            grouped_true,
            grouped_prob,
            threshold
        )

        result = {
            "group_size": group_size,
            "threshold": round(
                float(threshold),
                2
            ),
            "ROC_AUC": auc,
            **metrics
        }

        all_results.append(result)

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    group_results = [
        r for r in all_results
        if r["group_size"] == group_size
    ]

    best = max(
        group_results,
        key=lambda x: x["f1"]
    )

    print("\nBest F1 threshold:")
    print(
        "Threshold:",
        best["threshold"]
    )

    print(
        "Accuracy:",
        f"{best['accuracy']:.3f}"
    )

    print(
        "Sensitivity:",
        f"{best['sensitivity']:.3f}"
    )

    print(
        "Specificity:",
        f"{best['specificity']:.3f}"
    )

    print(
        "F1:",
        f"{best['f1']:.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)

# ============================================================
# SUMMARY
# ============================================================

print("\nSUMMARY")
print("-" * 70)

for group_size in GROUP_SIZES:

    subset = results_df[
        results_df["group_size"] == group_size
    ]

    best = subset.loc[
        subset["f1"].idxmax()
    ]

    print(
        f"\n{group_size} image(s):"
    )

    print(
        f"  ROC-AUC:     {best['ROC_AUC']:.4f}"
    )

    print(
        f"  Threshold:   {best['threshold']:.2f}"
    )

    print(
        f"  Accuracy:    {best['accuracy']:.3f}"
    )

    print(
        f"  Sensitivity: {best['sensitivity']:.3f}"
    )

    print(
        f"  Specificity: {best['specificity']:.3f}"
    )

    print(
        f"  F1:          {best['f1']:.3f}"
    )