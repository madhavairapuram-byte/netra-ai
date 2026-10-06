from pathlib import Path
import copy
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "clean_multiclass"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42

MODEL_PATH = MODEL_DIR / "eyecheck_multiclass_efficientnet_b0_v1.pth"
REPORT_PATH = REPORT_DIR / "multiclass_training_results.txt"
CLASS_PATH = REPORT_DIR / "multiclass_class_names.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("EYECHECK AI - MULTICLASS TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.03
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=val_transform
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print()

print(f"Number of classes: {num_classes}")

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")

print()

with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "classes": class_names,
            "class_to_index": train_dataset.class_to_idx
        },
        f,
        indent=4
    )


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = [0] * num_classes

for _, label in train_dataset.samples:
    class_counts[label] += 1

print("Training class counts:")

for i, count in enumerate(class_counts):
    print(f"  {class_names[i]}: {count}")

print()


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

class_weights = []

for count in class_counts:
    class_weights.append(1.0 / count)

sample_weights = [
    class_weights[label]
    for _, label in train_dataset.samples
]

sample_weights = torch.DoubleTensor(sample_weights)

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print("Loading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

# Replace final classifier
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_samples += labels.size(0)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return (
        average_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
best_state = None
best_epoch = 0

history = []

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        total_samples += labels.size(0)

    train_loss = running_loss / total_samples

    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.7f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = epoch + 1


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")

model.load_state_dict(best_state)


# ============================================================
# VALIDATION RESULTS
# ============================================================

(
    val_loss,
    val_accuracy,
    val_labels,
    val_predictions
) = evaluate(
    model,
    val_loader
)


# ============================================================
# TEST RESULTS
# ============================================================

(
    test_loss,
    test_accuracy,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader
)


# ============================================================
# METRICS
# ============================================================

val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    val_labels,
    val_predictions,
    average="macro",
    zero_division=0
)

test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

test_cm = confusion_matrix(
    test_labels,
    test_predictions
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

test_report = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_index": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "input_resize": 56,
        "best_epoch": best_epoch
    },
    MODEL_PATH
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EYECHECK AI - MULTICLASS TRAINING REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Device: {DEVICE}\n"
    )

    if torch.cuda.is_available():
        f.write(
            f"GPU: {torch.cuda.get_device_name(0)}\n"
        )

    f.write("\n")

    f.write(
        f"Classes: {class_names}\n"
    )

    f.write(
        f"Training images: {len(train_dataset)}\n"
    )

    f.write(
        f"Validation images: {len(val_dataset)}\n"
    )

    f.write(
        f"Test images: {len(test_dataset)}\n\n"
    )

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation loss: {best_val_loss:.6f}\n\n"
    )

    f.write(
        "VALIDATION RESULTS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Loss: {val_loss:.6f}\n"
    )

    f.write(
        f"Accuracy: {val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro Precision: {val_precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {val_recall:.4f}\n"
    )

    f.write(
        f"Macro F1: {val_f1:.4f}\n\n"
    )

    f.write(
        "TEST RESULTS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Accuracy: {test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro Precision: {test_precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {test_recall:.4f}\n"
    )

    f.write(
        f"Macro F1: {test_f1:.4f}\n\n"
    )

    f.write(
        "TEST CONFUSION MATRIX\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(test_cm)
    )

    f.write("\n\n")

    f.write(
        "TEST CLASSIFICATION REPORT\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        test_report
    )

    f.write("\n\n")

    f.write(
        "TRAINING HISTORY\n"
    )

    f.write("-" * 70 + "\n")

    for item in history:

        f.write(
            f"Epoch {item['epoch']:02d}: "
            f"train_loss={item['train_loss']:.6f}, "
            f"val_loss={item['val_loss']:.6f}, "
            f"val_accuracy={item['val_accuracy']:.6f}, "
            f"lr={item['learning_rate']:.8f}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("MULTICLASS TRAINING COMPLETE")
print("=" * 70)

print()

print("Validation:")
print(f"  Accuracy:  {val_accuracy * 100:.2f}%")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1:        {val_f1:.4f}")

print()

print("Test:")
print(f"  Accuracy:  {test_accuracy * 100:.2f}%")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1:        {test_f1:.4f}")

print()

print("Model saved to:")
print(MODEL_PATH)

print()

print("Report saved to:")
print(REPORT_PATH)

print()

print("=" * 70)
print("DONE")
print("=" * 70)