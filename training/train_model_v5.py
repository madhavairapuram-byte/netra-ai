import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "data/clean_split/train"
VAL_DIR = "data/clean_split/val"

MODEL_PATH = "models/conjunctivitis_efficientnet_b0_v5.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 16

WARMUP_EPOCHS = 3
FINETUNE_EPOCHS = 20

WARMUP_LR = 1e-4
FINETUNE_LR = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 6

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MODEL V5 TRAINING")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.90, 1.10)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.02
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )
        ],
        p=0.15
    ),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.20,
        scale=(0.02, 0.12),
        ratio=(0.3, 3.3)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClasses:")
print(train_dataset.class_to_idx)

print(
    "\nTraining images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# CLASS BALANCING
# ============================================================

targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    targets
)

print("\nClass counts:")

for index, count in enumerate(
    class_counts
):
    print(
        index,
        train_dataset.classes[index],
        count
    )


class_weights = (
    len(targets) /
    (
        len(class_counts) *
        class_counts
    )
)

sample_weights = np.array([
    class_weights[label]
    for label in targets
])

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(
        sample_weights
    ),
    num_samples=len(
        sample_weights
    ),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

weights = (
    EfficientNet_B0_Weights.DEFAULT
)

model = models.efficientnet_b0(
    weights=weights
)

num_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model):

    model.eval()

    all_labels = []
    all_probabilities = []
    all_predictions = []

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    all_predictions = np.array(
        all_predictions
    )

    average_loss = (
        total_loss /
        len(val_dataset)
    )

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

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    return (
        average_loss,
        accuracy,
        precision,
        sensitivity,
        f1,
        auc
    )


# ============================================================
# STAGE 1 — WARMUP
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 — CLASSIFIER WARMUP")
print("=" * 70)


for parameter in model.features.parameters():
    parameter.requires_grad = False


optimizer = optim.AdamW(
    model.classifier.parameters(),
    lr=WARMUP_LR,
    weight_decay=WEIGHT_DECAY
)


best_auc = -np.inf

best_state = None


for epoch in range(
    WARMUP_EPOCHS
):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

    (
        val_loss,
        accuracy,
        precision,
        sensitivity,
        f1,
        auc
    ) = evaluate(model)

    print(
        f"Epoch {epoch + 1}/{WARMUP_EPOCHS} | "
        f"Train Loss: "
        f"{running_loss / len(train_dataset):.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {accuracy * 100:.2f}% | "
        f"AUC: {auc:.4f}"
    )

    if auc > best_auc:

        best_auc = auc

        best_state = copy.deepcopy(
            model.state_dict()
        )


model.load_state_dict(
    best_state
)


# ============================================================
# STAGE 2 — FULL FINE-TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 — FULL FINE-TUNING")
print("=" * 70)


for parameter in model.parameters():
    parameter.requires_grad = True


optimizer = optim.AdamW(
    model.parameters(),
    lr=FINETUNE_LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


best_auc = -np.inf

best_state = copy.deepcopy(
    model.state_dict()
)

epochs_without_improvement = 0


# ============================================================
# TRAIN
# ============================================================

for epoch in range(
    FINETUNE_EPOCHS
):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

    (
        val_loss,
        accuracy,
        precision,
        sensitivity,
        f1,
        auc
    ) = evaluate(model)

    scheduler.step(
        auc
    )

    print(
        f"Epoch {epoch + 1}/{FINETUNE_EPOCHS} | "
        f"Train Loss: "
        f"{running_loss / len(train_dataset):.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {accuracy * 100:.2f}% | "
        f"Precision: {precision:.3f} | "
        f"Sensitivity: {sensitivity:.3f} | "
        f"F1: {f1:.3f} | "
        f"AUC: {auc:.4f}"
    )

    if auc > best_auc:

        best_auc = auc

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        print(
            "  --> New best validation AUC"
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# SAVE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_to_idx":
            train_dataset.class_to_idx,

        "image_size":
            IMAGE_SIZE,

        "model_version":
            "V5",

        "best_validation_auc":
            best_auc
    },
    MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

(
    val_loss,
    accuracy,
    precision,
    sensitivity,
    f1,
    auc
) = evaluate(model)


print("\n")
print("=" * 70)
print("V5 FINAL VALIDATION")
print("=" * 70)

print(
    f"\nValidation loss: {val_loss:.4f}"
)

print(
    f"Accuracy:        {accuracy * 100:.2f}%"
)

print(
    f"Precision:       {precision:.4f}"
)

print(
    f"Sensitivity:     {sensitivity:.4f}"
)

print(
    f"F1-score:        {f1:.4f}"
)

print(
    f"ROC-AUC:         {auc:.4f}"
)

print(
    "\nBest validation AUC:",
    f"{best_auc:.4f}"
)

print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)

print(
    "\nExternal dataset was NOT used for training."
)