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
# PATH
# ============================================================

CSV_PATH = "reports/external_validation_v4_predictions.csv"

# ============================================================
# LOAD RESULTS
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find: {CSV_PATH}\n"
        "Make sure you are running this from the project folder."
    )

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("V4 THRESHOLD ANALYSIS")
print("=" * 70)

print("\nColumns found in prediction file:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# FIND ACTUAL LABEL COLUMN
# ============================================================

actual_candidates = [
    "actual",
    "actual_label",
    "true_label",
    "true",
    "label",
    "ground_truth",
    "ground_truth_label"
]

actual_col = None

for col in actual_candidates:
    if col in df.columns:
        actual_col = col
        break

if actual_col is None:
    raise ValueError(
        "\nCould not automatically find the actual-label column.\n"
        "Columns found:\n"
        + str(list(df.columns))
    )

# ============================================================
# FIND CONJUNCTIVITIS PROBABILITY COLUMN
# ============================================================

prob_candidates = [
    "conjunctivitis_probability",
    "prob_conjunctivitis",
    "conjunctivitis_prob",
    "probability_conjunctivitis",
    "conjunctivitis_score",
    "probability",
    "score"
]

prob_col = None

for col in prob_candidates:
    if col in df.columns:
        prob_col = col
        break

# If not found, search column names automatically
if prob_col is None:

    for col in df.columns:

        name = col.lower()

        if (
            "conj" in name
            and ("prob" in name or "score" in name)
        ):
            prob_col = col
            break

if prob_col is None:
    raise ValueError(
        "\nCould not automatically find the conjunctivitis probability column.\n"
        "Columns found:\n"
        + str(list(df.columns))
    )

print("\nUsing actual label column:")
print(actual_col)

print("\nUsing probability column:")
print(prob_col)

# ============================================================
# CONVERT LABELS
# ============================================================

def convert_actual_label(value):

    value = str(value).strip().lower()

    if value in ["conjunctivitis", "1", "infected", "positive"]:
        return 1

    if value in ["normal", "healthy", "0", "negative"]:
        return 0

    raise ValueError(
        f"Unknown actual label: {value}"
    )


y_true = df[actual_col].apply(convert_actual_label).values

# ============================================================
# CONVERT PROBABILITIES
# ============================================================

y_prob = pd.to_numeric(
    df[prob_col],
    errors="coerce"
).values

if np.isnan(y_prob).any():
    raise ValueError(
        "Some probability values could not be converted to numbers."
    )

# Handle probabilities accidentally stored as percentages
if y_prob.max() > 1:
    y_prob = y_prob / 100.0

# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(y_true, y_prob)

print("\nNumber of images:", len(y_true))
print("Normal:", np.sum(y_true == 0))
print("Conjunctivitis:", np.sum(y_true == 1))

print(f"\nROC-AUC: {auc:.4f}")

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.05,
    1.00,
    0.05
)

results = []

print("\n")
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Sensitivity':<14}"
    f"{'Specificity':<14}"
    f"{'F1':<12}"
)

print("-" * 70)

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

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

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.3f}"
        f"{precision:<12.3f}"
        f"{sensitivity:<14.3f}"
        f"{specificity:<14.3f}"
        f"{f1:<12.3f}"
    )

# ============================================================
# FIND BEST THRESHOLDS
# ============================================================

results_df = pd.DataFrame(results)

best_accuracy = results_df.loc[
    results_df["accuracy"].idxmax()
]

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_balanced = results_df.loc[
    (results_df["sensitivity"] + results_df["specificity"]).idxmax()
]

# ============================================================
# PRINT BEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BEST THRESHOLD BY ACCURACY")
print("=" * 70)

print(best_accuracy.to_string())

print("\n")
print("=" * 70)
print("BEST THRESHOLD BY F1-SCORE")
print("=" * 70)

print(best_f1.to_string())

print("\n")
print("=" * 70)
print("BEST THRESHOLD BY SENSITIVITY + SPECIFICITY")
print("=" * 70)

print(best_balanced.to_string())

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "reports/v4_threshold_analysis.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nThreshold results saved to:\n{output_path}"
)