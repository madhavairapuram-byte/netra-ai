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
# MODEL VERSION 4
# RESOLUTION-HARMONIZED TRAINING
# AUGMENTED NORMAL DATASET EXPERIMENT
# ============================================================

TRAIN_DIR = "data/clean_split/train"
VAL_DIR = "data/clean_split/val"

MODEL_DIR = "models"

# IMPORTANT:
# Save this experiment as a NEW model.
# The original V4 model will NOT be overwritten.
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "conjunctivitis_efficientnet_b0_v4_augmented_normal.pth"
)

IMAGE_SIZE = 224

# Every image is first reduced to this resolution.
# This removes the large resolution difference between
# normal and conjunctivitis images in the source dataset.
HARMONIZED_SIZE = 56

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 1e-4

SEED = 42

NUM_WORKERS = 0


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("MODEL V4 + AUGMENTED NORMAL DATASET")
print("RESOLUTION-HARMONIZED TRAINING")
print("=" * 70)

print()

print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    # --------------------------------------------------------
    # First standardize the image geometry.
    # --------------------------------------------------------

    transforms.Resize(
        (HARMONIZED_SIZE, HARMONIZED_SIZE)
    ),

    # --------------------------------------------------------
    # Upscale the standardized 56x56 image to 224x224.
    # --------------------------------------------------------

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # --------------------------------------------------------
    # Mild augmentation.
    # --------------------------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
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
    ),

    transforms.RandomErasing(
        p=0.20,
        scale=(0.02, 0.08),
        ratio=(0.5, 2.0)
    )
])


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (HARMONIZED_SIZE, HARMONIZED_SIZE)
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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
# LOAD DATASETS
# ============================================================

print("-" * 70)
print("LOADING DATASETS")
print("-" * 70)

print()

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print()

print(
    "Class mapping:"
)

for class_name, class_index in (
    train_dataset.class_to_idx.items()
):

    print(
        f"  {class_index} = {class_name}"
    )

print()


# ============================================================
# CLASS COUNTS
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets
)

print(
    "Training class counts:"
)

for class_index, count in enumerate(
    class_counts
):

    print(
        f"  Class {class_index}: {count}"
    )

print()


# ============================================================
# BALANCED SAMPLING
# ============================================================

class_weights = (
    1.0 / class_counts
)

sample_weights = (
    class_weights[
        train_targets
    ]
)

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

print("-" * 70)
print("LOADING EFFICIENTNET-B0")
print("-" * 70)

print()

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model = model.to(device)

print(
    "Model loaded successfully."
)

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
# SCHEDULER
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

best_model_weights = copy.deepcopy(
    model.state_dict()
)


print("=" * 70)
print("STARTING MODEL V4 + AUGMENTED NORMAL TRAINING")
print("=" * 70)

print()

for epoch in range(EPOCHS):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
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
            loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0
        * correct
        / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = (
        val_loss_total
        / val_total
    )

    val_accuracy = (
        100.0
        * val_correct
        / val_total
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(
        val_accuracy
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )

        best_model_weights = (
            copy.deepcopy(
                model.state_dict()
            )
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        marker = " <-- BEST"

    else:

        marker = ""


    # ========================================================
    # PRINT
    # ========================================================

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.2f}% | "
        f"Val Loss: "
        f"{val_loss:.4f} | "
        f"Val Acc: "
        f"{val_accuracy:.2f}% | "
        f"LR: "
        f"{current_lr:.2e}"
        f"{marker}"
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
print("MODEL V4 + AUGMENTED NORMAL TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.2f}%"
)

print()

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()

print(
    "External validation dataset was NOT used."
)

print()

print(
    "Resolution harmonization:"
)

print(
    f"All images reduced to "
    f"{HARMONIZED_SIZE}x{HARMONIZED_SIZE}"
    f" before being resized to "
    f"{IMAGE_SIZE}x{IMAGE_SIZE}."
)

print()

print(
    "Original V4 model was NOT overwritten."
)

print(
    "New model:"
)

print(
    "conjunctivitis_efficientnet_b0_v4_augmented_normal.pth"
)

print("=" * 70)