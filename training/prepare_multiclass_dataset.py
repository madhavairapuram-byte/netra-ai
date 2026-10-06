import os
import shutil
import random

from PIL import Image
import imagehash


# ============================================================
# MULTICLASS LEAKAGE-RESISTANT DATASET PREPARATION
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    ROOT,
    "data",
    "raw"
)

CLEAN_DIR = os.path.join(
    ROOT,
    "data",
    "clean_multiclass"
)


# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "normal",
    "conjunctivitis",
    "Cataract",
    "stye",
    "Uveitis"
]

RANDOM_SEED = 42

HASH_THRESHOLD = 6

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)

print(
    "CREATING MULTICLASS LEAKAGE-RESISTANT DATASET"
)

print("=" * 70)

print()

print(
    "Classes:"
)

for class_name in CLASSES:

    print(
        f"  - {class_name}"
    )

print()


# ============================================================
# CHECK RAW DATA
# ============================================================

print("=" * 70)

print(
    "CHECKING RAW DATA"
)

print("=" * 70)

print()


all_images = {}


for class_name in CLASSES:

    class_dir = os.path.join(
        RAW_DIR,
        class_name
    )

    if not os.path.isdir(
        class_dir
    ):

        raise FileNotFoundError(
            f"Missing class folder:\n{class_dir}"
        )


    files = []

    for filename in os.listdir(
        class_dir
    ):

        filepath = os.path.join(
            class_dir,
            filename
        )

        if not os.path.isfile(
            filepath
        ):
            continue

        try:

            with Image.open(
                filepath
            ):

                files.append(
                    filepath
                )

        except Exception:

            print(
                "Skipping invalid image:",
                filepath
            )


    all_images[class_name] = files


    print(
        f"{class_name}: "
        f"{len(files)} images"
    )


print()


# ============================================================
# REMOVE OLD CLEAN DATASET
# ============================================================

print("=" * 70)

print(
    "REMOVING OLD MULTICLASS CLEAN DATASET"
)

print("=" * 70)

print()


if os.path.exists(
    CLEAN_DIR
):

    print(
        "Deleting:"
    )

    print(
        CLEAN_DIR
    )

    shutil.rmtree(
        CLEAN_DIR
    )

    print(
        "Old multiclass dataset removed."
    )

else:

    print(
        "No previous multiclass dataset found."
    )


print()


# ============================================================
# HASH FUNCTION
# ============================================================

def calculate_hash(
    filepath
):

    try:

        with Image.open(
            filepath
        ) as image:

            image = image.convert(
                "RGB"
            )

            return imagehash.phash(
                image
            )

    except Exception as e:

        print(
            "Hash error:",
            filepath,
            e
        )

        return None


# ============================================================
# FIND SIMILARITY GROUPS
# ============================================================

def create_similarity_groups(
    image_files
):

    hashes = []

    print(
        f"Calculating hashes for "
        f"{len(image_files)} images..."
    )


    for index, filepath in enumerate(
        image_files,
        start=1
    ):

        image_hash = calculate_hash(
            filepath
        )

        if image_hash is not None:

            hashes.append(
                (
                    filepath,
                    image_hash
                )
            )


        if (
            index % 100 == 0
            or index == len(image_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(image_files)}"
            )


    print()

    print(
        "Comparing images for near-duplicates..."
    )


    n = len(hashes)

    parent = list(
        range(n)
    )


    def find(
        x
    ):

        while parent[x] != x:

            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x


    def union(
        a,
        b
    ):

        root_a = find(
            a
        )

        root_b = find(
            b
        )

        if root_a != root_b:

            parent[root_b] = root_a


    matches = 0


    for i in range(
        n
    ):

        for j in range(
            i + 1,
            n
        ):

            distance = (
                hashes[i][1]
                - hashes[j][1]
            )


            if (
                distance
                <= HASH_THRESHOLD
            ):

                union(
                    i,
                    j
                )

                matches += 1


    groups = {}


    for index in range(
        n
    ):

        root = find(
            index
        )

        if root not in groups:

            groups[root] = []

        groups[root].append(
            hashes[index][0]
        )


    similarity_groups = list(
        groups.values()
    )


    print()

    print(
        f"Similarity matches: "
        f"{matches}"
    )

    print(
        f"Similarity groups: "
        f"{len(similarity_groups)}"
    )

    print()

    return similarity_groups


# ============================================================
# SPLIT GROUPS
# ============================================================

def split_groups(
    groups
):

    random.shuffle(
        groups
    )


    total_images = sum(
        len(group)
        for group in groups
    )


    target_train = (
        total_images
        * TRAIN_RATIO
    )

    target_val = (
        total_images
        * VAL_RATIO
    )


    train_groups = []

    val_groups = []

    test_groups = []


    train_count = 0

    val_count = 0


    for group in groups:

        group_size = len(
            group
        )


        if train_count < target_train:

            train_groups.append(
                group
            )

            train_count += group_size


        elif val_count < target_val:

            val_groups.append(
                group
            )

            val_count += group_size


        else:

            test_groups.append(
                group
            )


    train_files = [
        filepath
        for group in train_groups
        for filepath in group
    ]


    val_files = [
        filepath
        for group in val_groups
        for filepath in group
    ]


    test_files = [
        filepath
        for group in test_groups
        for filepath in group
    ]


    return (
        train_files,
        val_files,
        test_files
    )


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in CLASSES:

    print("=" * 70)

    print(
        f"PROCESSING: "
        f"{class_name.upper()}"
    )

    print("=" * 70)

    print()


    image_files = all_images[
        class_name
    ]


    groups = create_similarity_groups(
        image_files
    )


    print(
        f"Total images: "
        f"{len(image_files)}"
    )

    print(
        f"Similarity groups: "
        f"{len(groups)}"
    )

    print()


    (
        train_files,
        val_files,
        test_files
    ) = split_groups(
        groups
    )


    print(
        "SPLIT"
    )

    print(
        f"Training:   "
        f"{len(train_files)}"
    )

    print(
        f"Validation: "
        f"{len(val_files)}"
    )

    print(
        f"Test:       "
        f"{len(test_files)}"
    )

    print(
        f"Total:      "
        f"{len(train_files) + len(val_files) + len(test_files)}"
    )

    print()


    # --------------------------------------------------------
    # CREATE CLASS DIRECTORIES
    # --------------------------------------------------------

    train_dir = os.path.join(
        CLEAN_DIR,
        "train",
        class_name
    )

    val_dir = os.path.join(
        CLEAN_DIR,
        "val",
        class_name
    )

    test_dir = os.path.join(
        CLEAN_DIR,
        "test",
        class_name
    )


    os.makedirs(
        train_dir,
        exist_ok=True
    )

    os.makedirs(
        val_dir,
        exist_ok=True
    )

    os.makedirs(
        test_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    def copy_files(
        files,
        destination
    ):

        for index, source in enumerate(
            files
        ):

            filename = os.path.basename(
                source
            )

            destination_path = os.path.join(
                destination,
                filename
            )


            # ------------------------------------------------
            # Prevent filename collision.
            # ------------------------------------------------

            if os.path.exists(
                destination_path
            ):

                base, extension = os.path.splitext(
                    filename
                )

                counter = 1

                while os.path.exists(
                    destination_path
                ):

                    new_filename = (
                        f"{base}_"
                        f"{counter}"
                        f"{extension}"
                    )

                    destination_path = os.path.join(
                        destination,
                        new_filename
                    )

                    counter += 1


            shutil.copy2(
                source,
                destination_path
            )


    copy_files(
        train_files,
        train_dir
    )

    copy_files(
        val_files,
        val_dir
    )

    copy_files(
        test_files,
        test_dir
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)

print(
    "MULTICLASS CLEAN DATASET CREATED"
)

print("=" * 70)

print()

print(
    "Location:"
)

print(
    CLEAN_DIR
)

print()

print(
    "RAW DATA:"
)

total_raw = 0

for class_name in CLASSES:

    count = len(
        all_images[class_name]
    )

    total_raw += count

    print(
        f"{class_name}: {count}"
    )


print(
    f"Total: {total_raw}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "- Similar images were kept in the same split."
)

print(
    "- The raw dataset was NOT changed."
)

print(
    "- The previous multiclass clean split was rebuilt."
)

print(
    f"- Random seed: {RANDOM_SEED}"
)

print(
    f"- Hash threshold: {HASH_THRESHOLD}"
)

print()

print(
    "DONE"
)

print("=" * 70)