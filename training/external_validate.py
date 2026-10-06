import os
import csv
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    "models/"
    "conjunctivitis_efficientnet_b0_v4_augmented_normal.pth"
)

EXTERNAL_DIR = "data/external"

OUTPUT_CSV = (
    "reports/"
    "external_validation_v4_augmented_normal_predictions.csv"
)

OUTPUT_RESULTS = (
    "reports/"
    "external_validation_v4_augmented_normal_results.txt"
)

# Known duplicate between the external dataset and the
# original conjunctivitis dataset.
EXCLUDED_FILE = "170.jpeg"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MODEL V4 + AUGMENTED NORMAL - EXTERNAL VALIDATION")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = models.efficientnet_b0(
    weights=None
)

num_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    num_features,
    2
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)

model.eval()


print(
    "\nModel loaded successfully:"
)

print(
    MODEL_PATH
)


# ============================================================
# EXTERNAL DATA
# ============================================================

class_folders = {
    "healthy_eye": "normal",
    "infected_eye": "conjunctivitis"
}


image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# PREDICTION
# ============================================================

rows = []

y_true = []

y_prob = []

y_pred = []


for folder_name, true_label in class_folders.items():

    folder_path = os.path.join(
        EXTERNAL_DIR,
        folder_name
    )

    if not os.path.exists(folder_path):

        raise FileNotFoundError(
            f"Folder not found:\n{folder_path}"
        )

    for filename in sorted(
        os.listdir(folder_path)
    ):

        if not filename.lower().endswith(
            image_extensions
        ):
            continue

        # ----------------------------------------------------
        # Exclude known duplicate
        # ----------------------------------------------------

        if (
            true_label == "conjunctivitis"
            and filename.lower()
            == EXCLUDED_FILE.lower()
        ):

            print(
                "Excluded known duplicate:",
                filename
            )

            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image_tensor = transform(
                image
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(
                    image_tensor
                )

                # IMPORTANT:
                # ImageFolder class ordering is:
                #
                # 0 = conjunctivitis
                # 1 = normal
                #
                # Therefore probability [0, 0]
                # is the conjunctivitis probability.

                probability = torch.softmax(
                    output,
                    dim=1
                )[0, 0].item()


            prediction = (
                "conjunctivitis"
                if probability >= 0.5
                else "normal"
            )


            true_binary = (
                1
                if true_label == "conjunctivitis"
                else 0
            )

            predicted_binary = (
                1
                if prediction == "conjunctivitis"
                else 0
            )


            y_true.append(
                true_binary
            )

            y_prob.append(
                probability
            )

            y_pred.append(
                predicted_binary
            )


            rows.append({

                "filename":
                    filename,

                "true_label":
                    true_label,

                "predicted_label":
                    prediction,

                "conjunctivitis_probability":
                    probability

            })


        except Exception as e:

            print(
                "Error processing:",
                image_path,
                e
            )


# ============================================================
# METRICS
# ============================================================

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

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1
    ]
)

tn, fp, fn, tp = cm.ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


report = classification_report(
    y_true,
    y_pred,

    target_names=[
        "Normal",
        "Conjunctivitis"
    ],

    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "V4 + AUGMENTED NORMAL"
)

print(
    "EXTERNAL VALIDATION RESULTS"
)

print("=" * 70)


print(
    f"\nImages tested: {len(y_true)}"
)


print(
    "Healthy:",
    sum(
        1
        for x in y_true
        if x == 0
    )
)


print(
    "Conjunctivitis:",
    sum(
        1
        for x in y_true
        if x == 1
    )
)


print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)


print(
    f"Sensitivity: "
    f"{sensitivity * 100:.2f}%"
)


print(
    f"Specificity: "
    f"{specificity * 100:.2f}%"
)


print(
    f"F1-score: "
    f"{f1 * 100:.2f}%"
)


print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


print(
    "\nClassification Report:"
)

print(
    report
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

os.makedirs(
    "reports",
    exist_ok=True
)


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,

        fieldnames=[
            "filename",
            "true_label",
            "predicted_label",
            "conjunctivitis_probability"
        ]
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_RESULTS,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MODEL V4 + AUGMENTED NORMAL "
        "- EXTERNAL VALIDATION RESULTS\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Images tested: "
        f"{len(y_true)}\n"
    )

    f.write(
        f"Healthy: "
        f"{sum(1 for x in y_true if x == 0)}\n"
    )

    f.write(
        f"Conjunctivitis: "
        f"{sum(1 for x in y_true if x == 1)}\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{precision * 100:.2f}%\n"
    )

    f.write(
        f"Sensitivity: "
        f"{sensitivity * 100:.2f}%\n"
    )

    f.write(
        f"Specificity: "
        f"{specificity * 100:.2f}%\n"
    )

    f.write(
        f"F1-score: "
        f"{f1 * 100:.2f}%\n"
    )

    f.write(
        f"ROC-AUC: "
        f"{roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
        + "\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")

print(
    "Predictions saved to:"
)

print(
    OUTPUT_CSV
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_RESULTS
)


print(
    "\nExternal dataset was NOT used for training."
)

print(
    "The known duplicate 170.jpeg was excluded."
)

print("=" * 70)