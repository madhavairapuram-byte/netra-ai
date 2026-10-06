import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "data/clean_split/train"
VAL_DIR = "data/clean_split/val"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "conjunctivitis_efficientnet_b0_v3.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

SEED = 42

NUM_WORKERS = 0

os.makedirs(MODEL_DIR, exist_ok=True)


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
print("MODEL VERSION 3")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    # Randomly crop and resize the eye image.
    # This reduces dependence on exact image composition.
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.00),
        ratio=(0.90, 1.10)
    ),

    # Eyes can appear horizontally mirrored without
    # changing the disease class.
    transforms.RandomHorizontalFlip(p=0.5),

    # Small rotations simulate different phone/camera angles.
    transforms.RandomRotation(
        degrees=10
    ),

    # Moderate lighting variation.
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),

    transforms.ToTensor(),

    # ImageNet normalization for pretrained EfficientNet.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    # Hide small portions of the image occasionally.
    # This discourages dependence on one tiny visual cue.
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.10),
        ratio=(0.5, 2.0)
    )
])


# Validation data must NOT be augmented.
val_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

print("Loading datasets...")
print()

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print()
print("Class mapping:")

for class_name, class_index in train_dataset.class_to_idx.items():
    print(f"  {class_index} = {class_name}")

print()


# ============================================================
# CLASS-BALANCED SAMPLING
# ============================================================

# The clean training set contains more normal images than
# conjunctivitis images. We compensate for this during training.

train_targets = np.array(train_dataset.targets)

class_counts = np.bincount(train_targets)

print("Training class counts:")

for class_index, count in enumerate(class_counts):
    print(f"  Class {class_index}: {count}")

print()

class_weights = 1.0 / class_counts

sample_weights = class_weights[train_targets]

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print("Loading pretrained EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

# Replace final classifier with binary classifier.
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(device)

print("Model loaded.")
print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = val_loss_total / val_total

    val_accuracy = (
        100.0 * val_correct / val_total
    )

    scheduler.step(val_accuracy)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        best_marker = "  <-- BEST"

    else:

        best_marker = ""


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"LR: {current_lr:.2e}"
        f"{best_marker}"
    )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("MODEL VERSION 3 TRAINING COMPLETE")
print("=" * 70)

print()
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print()
print("Model saved to:")

print(MODEL_PATH)

print()
print("IMPORTANT:")
print("The external validation dataset was NOT used during training.")

print("=" * 70)