from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data" / "clean_split" / "test"

MODEL_PATH = (
    ROOT
    / "models"
    / "conjunctivitis_efficientnet_b0_clean.pth"
)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 2


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CONJUNCTIVITIS AI - MODEL VERSION 2 TEST")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# =========================================================
# TEST IMAGE TRANSFORM
# =========================================================

test_transforms = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD TEST DATASET
# =========================================================

print("Loading clean test dataset...")

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print("Classes:")
print(test_dataset.class_to_idx)

print()
print("Test images:", len(test_dataset))


# =========================================================
# CREATE MODEL
# =========================================================

print()
print("Loading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# =========================================================
# REPLACE CLASSIFIER
# =========================================================

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


# =========================================================
# LOAD OUR TRAINED MODEL
# =========================================================

print()
print("Loading trained model:")

print(MODEL_PATH)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# =========================================================
# RUN TEST
# =========================================================

all_labels = []
all_predictions = []
all_probabilities = []


print()
print("Running test...")
print()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of conjunctivitis
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# =========================================================
# CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

sensitivity = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

tn, fp, fn, tp = cm.ravel()


# =========================================================
# SPECIFICITY
# =========================================================

if (tn + fp) > 0:

    specificity = tn / (tn + fp)

else:

    specificity = 0.0


# =========================================================
# PRINT FINAL RESULTS
# =========================================================

print("=" * 70)
print("FINAL CLEAN TEST RESULTS")
print("=" * 70)

print()

print(
    f"Accuracy:     {accuracy * 100:.2f}%"
)

print(
    f"Precision:    {precision * 100:.2f}%"
)

print(
    f"Sensitivity:  {sensitivity * 100:.2f}%"
)

print(
    f"Specificity:  {specificity * 100:.2f}%"
)

print(
    f"F1-score:     {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:      {roc_auc:.4f}"
)

print()

print("Confusion Matrix:")

print(cm)

print()

print("Detailed Classification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Normal",
            "Conjunctivitis"
        ],
        zero_division=0
    )
)

print("=" * 70)

print()
print("MODEL VERSION 2 TEST COMPLETE")

print("=" * 70)