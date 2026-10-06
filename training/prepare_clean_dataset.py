from pathlib import Path
import shutil
import random
from PIL import Image
import imagehash


# =========================================================
# SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
CLEAN_SPLIT_DIR = ROOT / "data" / "clean_split"

RANDOM_SEED = 42

# Images with perceptual hash distance <= this value
# are treated as belonging to the same similarity group.
HASH_THRESHOLD = 6

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# START
# =========================================================

random.seed(RANDOM_SEED)

print("=" * 70)
print("CREATING NEW LEAKAGE-RESISTANT DATASET SPLIT")
print("=" * 70)


# =========================================================
# FIND RAW IMAGES
# =========================================================

classes = [
    "normal",
    "conjunctivitis",
]

all_images = {}

for class_name in classes:

    folder = RAW_DIR / class_name

    if not folder.exists():

        print()
        print(f"ERROR: Folder does not exist:")
        print(folder)

        raise SystemExit


    files = sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    all_images[class_name] = files

    print(
        f"{class_name}: {len(files)} images"
    )


# =========================================================
# SAFETY CHECK
# =========================================================

expected_normal = 1048
expected_conjunctivitis = 357

if len(all_images["normal"]) != expected_normal:

    print()
    print("=" * 70)
    print("SAFETY STOP")
    print("=" * 70)

    print()
    print(
        f"Expected {expected_normal} normal images."
    )

    print(
        f"Found {len(all_images['normal'])}."
    )

    print()
    print("The dataset will NOT be prepared.")

    raise SystemExit


if len(all_images["conjunctivitis"]) != expected_conjunctivitis:

    print()
    print("=" * 70)
    print("SAFETY STOP")
    print("=" * 70)

    print()
    print(
        f"Expected {expected_conjunctivitis} conjunctivitis images."
    )

    print(
        f"Found {len(all_images['conjunctivitis'])}."
    )

    print()
    print("The dataset will NOT be prepared.")

    raise SystemExit


# =========================================================
# PERCEPTUAL HASH
# =========================================================

def calculate_hash(image_path):

    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            return imagehash.phash(image)

    except Exception as error:

        print()
        print("Could not process:")
        print(image_path)

        print("Error:")
        print(error)

        return None


# =========================================================
# CREATE SIMILARITY GROUPS
# =========================================================

def create_groups(files):

    hashes = {}

    print()
    print(
        f"Calculating hashes for {len(files)} images..."
    )

    for index, file in enumerate(files, start=1):

        image_hash = calculate_hash(file)

        if image_hash is not None:

            hashes[file] = image_hash

        if index % 100 == 0 or index == len(files):

            print(
                f"Processed {index}/{len(files)}"
            )


    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    if len(hashes) != len(files):

        print()
        print("ERROR:")
        print(
            f"Only {len(hashes)} of {len(files)} images "
            "could be processed."
        )

        raise SystemExit


    # -----------------------------------------------------
    # Union-Find
    # -----------------------------------------------------

    parent = {
        file: file
        for file in hashes
    }


    def find(x):

        while parent[x] != x:

            parent[x] = parent[parent[x]]

            x = parent[x]

        return x


    def union(a, b):

        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:

            parent[root_b] = root_a


    # -----------------------------------------------------
    # Compare every pair
    # -----------------------------------------------------

    files_list = list(hashes.keys())

    comparison_count = 0

    match_count = 0

    print()
    print("Comparing images for near-duplicates...")

    for i in range(len(files_list)):

        file_a = files_list[i]

        hash_a = hashes[file_a]

        for j in range(i + 1, len(files_list)):

            file_b = files_list[j]

            hash_b = hashes[file_b]

            distance = hash_a - hash_b

            comparison_count += 1

            if distance <= HASH_THRESHOLD:

                union(
                    file_a,
                    file_b
                )

                match_count += 1


    # -----------------------------------------------------
    # Build groups
    # -----------------------------------------------------

    groups = {}

    for file in files_list:

        root = find(file)

        if root not in groups:

            groups[root] = []

        groups[root].append(file)


    groups = list(groups.values())


    print()
    print(
        f"Similarity matches: {match_count}"
    )

    print(
        f"Similarity groups: {len(groups)}"
    )

    return groups


# =========================================================
# REMOVE OLD CLEAN SPLIT
# =========================================================

print()
print("=" * 70)
print("REMOVING OLD CLEAN SPLIT")
print("=" * 70)

if CLEAN_SPLIT_DIR.exists():

    print()
    print(
        f"Deleting old clean split:"
    )

    print(CLEAN_SPLIT_DIR)

    shutil.rmtree(CLEAN_SPLIT_DIR)

    print()
    print("Old clean split removed.")

else:

    print()
    print("No previous clean split found.")


# =========================================================
# CREATE NEW SPLIT
# =========================================================

for class_name in classes:

    print()
    print("=" * 70)
    print(
        f"PROCESSING: {class_name.upper()}"
    )
    print("=" * 70)

    files = all_images[class_name]

    groups = create_groups(files)

    print()
    print(
        f"Total images: {len(files)}"
    )

    print(
        f"Similarity groups: {len(groups)}"
    )


    # -----------------------------------------------------
    # Shuffle groups, NOT individual images
    # -----------------------------------------------------

    random.shuffle(groups)


    total_images = len(files)

    target_train = total_images * 0.70

    target_val = total_images * 0.15


    train_groups = []

    val_groups = []

    test_groups = []


    train_count = 0

    val_count = 0

    test_count = 0


    # -----------------------------------------------------
    # Assign complete groups
    # -----------------------------------------------------

    for group in groups:

        group_size = len(group)


        if train_count < target_train:

            train_groups.append(group)

            train_count += group_size


        elif val_count < target_val:

            val_groups.append(group)

            val_count += group_size


        else:

            test_groups.append(group)

            test_count += group_size


    # -----------------------------------------------------
    # Flatten
    # -----------------------------------------------------

    train_files = [
        image
        for group in train_groups
        for image in group
    ]

    val_files = [
        image
        for group in val_groups
        for image in group
    ]

    test_files = [
        image
        for group in test_groups
        for image in group
    ]


    # -----------------------------------------------------
    # Print split
    # -----------------------------------------------------

    print()
    print("SPLIT")

    print(
        f"Training:   {len(train_files)}"
    )

    print(
        f"Validation: {len(val_files)}"
    )

    print(
        f"Test:       {len(test_files)}"
    )

    print(
        f"Total:      "
        f"{len(train_files) + len(val_files) + len(test_files)}"
    )


    # -----------------------------------------------------
    # Create directories
    # -----------------------------------------------------

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        output_dir = (
            CLEAN_SPLIT_DIR
            / split_name
            / class_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    # -----------------------------------------------------
    # Copy images
    # -----------------------------------------------------

    for image in train_files:

        shutil.copy2(
            image,
            CLEAN_SPLIT_DIR
            / "train"
            / class_name
            / image.name
        )


    for image in val_files:

        shutil.copy2(
            image,
            CLEAN_SPLIT_DIR
            / "val"
            / class_name
            / image.name
        )


    for image in test_files:

        shutil.copy2(
            image,
            CLEAN_SPLIT_DIR
            / "test"
            / class_name
            / image.name
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("NEW CLEAN DATASET CREATED")
print("=" * 70)

print()
print("Location:")
print(CLEAN_SPLIT_DIR)

print()
print("RAW DATA:")
print(
    f"Normal: {len(all_images['normal'])}"
)

print(
    f"Conjunctivitis: "
    f"{len(all_images['conjunctivitis'])}"
)

print(
    f"Total: "
    f"{len(all_images['normal']) + len(all_images['conjunctivitis'])}"
)

print()
print("IMPORTANT:")
print("- Similar images were kept in the same split.")
print("- The raw dataset was NOT changed.")
print("- The previous clean split was completely rebuilt.")
print("- Random seed:", RANDOM_SEED)

print()
print("=" * 70)
print("DONE")
print("=" * 70)