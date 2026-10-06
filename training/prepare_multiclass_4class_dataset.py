from pathlib import Path
import shutil
import random

from PIL import Image
import imagehash


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean_multiclass_4class"


# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "normal",
    "conjunctivitis",
    "Cataract",
    "stye"
]

SEED = 42

HASH_THRESHOLD = 6

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)


# ============================================================
# FUNCTIONS
# ============================================================

def get_image_files(folder):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def calculate_hash(path):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img)
    except Exception:
        return None


def build_similarity_groups(files):
    """
    Groups visually similar images together so that
    near-duplicates stay in the same split.
    """

    hashes = {}

    for path in files:

        h = calculate_hash(path)

        if h is not None:
            hashes[path] = h

    groups = []
    assigned = set()

    for path in files:

        if path in assigned:
            continue

        group = [path]

        assigned.add(path)

        if path in hashes:

            for other in files:

                if other in assigned:
                    continue

                if other not in hashes:
                    continue

                distance = hashes[path] - hashes[other]

                if distance <= HASH_THRESHOLD:

                    group.append(other)
                    assigned.add(other)

        groups.append(group)

    return groups


def split_groups(groups):

    random.shuffle(groups)

    total_images = sum(
        len(group)
        for group in groups
    )

    target_train = int(
        total_images * TRAIN_RATIO
    )

    target_val = int(
        total_images * VAL_RATIO
    )

    train = []
    val = []
    test = []

    train_count = 0
    val_count = 0

    for group in groups:

        group_size = len(group)

        if train_count + group_size <= target_train:
            train.extend(group)
            train_count += group_size

        elif val_count + group_size <= target_val:
            val.extend(group)
            val_count += group_size

        else:
            test.extend(group)

    return train, val, test


def copy_files(files, destination):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(files):

        extension = source.suffix.lower()

        destination_file = (
            destination /
            f"{index:05d}{extension}"
        )

        shutil.copy2(
            source,
            destination_file
        )


# ============================================================
# CLEAR OLD DATASET
# ============================================================

if CLEAN_DIR.exists():

    print("Removing previous 4-class clean dataset...")

    shutil.rmtree(CLEAN_DIR)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    for class_name in CLASSES:

        (
            CLEAN_DIR /
            split /
            class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS DATA
# ============================================================

print("=" * 70)
print("PREPARING 4-CLASS MULTICLASS DATASET")
print("=" * 70)

print()

print("Classes:")

for class_name in CLASSES:
    print(f"  - {class_name}")

print()

grand_total = 0

grand_train = 0
grand_val = 0
grand_test = 0


for class_name in CLASSES:

    source_dir = RAW_DIR / class_name

    files = get_image_files(source_dir)

    print("-" * 70)

    print(
        f"{class_name}: {len(files)} raw images"
    )

    # --------------------------------------------------------
    # Build similarity groups
    # --------------------------------------------------------

    groups = build_similarity_groups(files)

    print(
        f"Similarity groups: {len(groups)}"
    )

    # --------------------------------------------------------
    # Split groups
    # --------------------------------------------------------

    train_files, val_files, test_files = split_groups(
        groups
    )

    print(
        f"Train: {len(train_files)}"
    )

    print(
        f"Val:   {len(val_files)}"
    )

    print(
        f"Test:  {len(test_files)}"
    )

    # --------------------------------------------------------
    # Copy files
    # --------------------------------------------------------

    copy_files(
        train_files,
        CLEAN_DIR / "train" / class_name
    )

    copy_files(
        val_files,
        CLEAN_DIR / "val" / class_name
    )

    copy_files(
        test_files,
        CLEAN_DIR / "test" / class_name
    )

    grand_total += len(files)

    grand_train += len(train_files)
    grand_val += len(val_files)
    grand_test += len(test_files)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("4-CLASS MULTICLASS DATASET CREATED")
print("=" * 70)

print()

print("Location:")
print(CLEAN_DIR)

print()

print("Classes included:")

for class_name in CLASSES:
    print(f"  {class_name}")

print()

print(
    f"Total raw images used: {grand_total}"
)

print(
    f"Train: {grand_train}"
)

print(
    f"Validation: {grand_val}"
)

print(
    f"Test: {grand_test}"
)

print()

print("Uveitis was NOT included.")

print("The original raw dataset was NOT modified.")

print()

print("=" * 70)
print("DONE")
print("=" * 70)