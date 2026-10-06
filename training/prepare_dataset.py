from pathlib import Path
import random
import shutil

# -----------------------------
# SETTINGS
# -----------------------------

SOURCE = Path("data/raw")
OUTPUT = Path("data/split")

CLASSES = ["normal", "conjunctivitis"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# -----------------------------
# CHECK SETTINGS
# -----------------------------

if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:
    raise ValueError("The split ratios must add up to 1.0")

# -----------------------------
# CREATE FOLDERS
# -----------------------------

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        folder = OUTPUT / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# SPLIT IMAGES
# -----------------------------

for class_name in CLASSES:

    source_folder = SOURCE / class_name

    images = [
        file for file in source_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    print(f"\n{class_name}: {len(images)} images found")

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"  Training:   {len(train_images)}")
    print(f"  Validation: {len(val_images)}")
    print(f"  Test:       {len(test_images)}")

    # Copy training images
    for image in train_images:
        destination = OUTPUT / "train" / class_name / image.name
        shutil.copy2(image, destination)

    # Copy validation images
    for image in val_images:
        destination = OUTPUT / "val" / class_name / image.name
        shutil.copy2(image, destination)

    # Copy test images
    for image in test_images:
        destination = OUTPUT / "test" / class_name / image.name
        shutil.copy2(image, destination)

print("\n--------------------------------")
print("Dataset preparation complete!")
print("--------------------------------")