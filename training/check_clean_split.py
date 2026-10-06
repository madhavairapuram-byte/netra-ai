from pathlib import Path

from PIL import Image
import imagehash


# ============================================================
# MULTICLASS CROSS-SPLIT NEAR-DUPLICATE CHECK
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = (
    ROOT
    / "data"
    / "clean_multiclass"
)

THRESHOLD = 6

SPLITS = [
    "train",
    "val",
    "test"
]

CLASSES = [
    "normal",
    "conjunctivitis",
    "Cataract",
    "stye",
    "Uveitis"
]

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "CHECKING MULTICLASS DATASET FOR "
    "CROSS-SPLIT NEAR-DUPLICATES"
)

print("=" * 70)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if not SPLIT_DIR.exists():

    raise FileNotFoundError(
        f"Clean multiclass dataset not found:\n"
        f"{SPLIT_DIR}"
    )


images = []


# ============================================================
# FIND IMAGES
# ============================================================

for split in SPLITS:

    for class_name in CLASSES:

        folder = (
            SPLIT_DIR
            / split
            / class_name
        )


        if not folder.exists():

            raise FileNotFoundError(
                f"Missing folder:\n{folder}"
            )


        files = [

            f

            for f in folder.iterdir()

            if (
                f.is_file()
                and
                f.suffix.lower()
                in IMAGE_EXTENSIONS
            )

        ]


        print(
            f"{split:5} | "
            f"{class_name:15} | "
            f"{len(files)} images"
        )


        for file in files:

            try:

                image = (
                    Image.open(file)
                    .convert("RGB")
                )


                hash_value = (
                    imagehash.phash(image)
                )


                images.append({

                    "split":
                        split,

                    "class":
                        class_name,

                    "path":
                        file,

                    "hash":
                        hash_value

                })


            except Exception as e:

                print(
                    f"Could not read "
                    f"{file}: {e}"
                )


print()

print(
    f"Total images checked: "
    f"{len(images)}"
)

print()


# ============================================================
# COMPARE IMAGES
# ============================================================

duplicates = []


for i in range(
    len(images)
):

    for j in range(
        i + 1,
        len(images)
    ):

        image_a = images[i]

        image_b = images[j]


        # ----------------------------------------------------
        # Only interested in different splits.
        # ----------------------------------------------------

        if (
            image_a["split"]
            ==
            image_b["split"]
        ):

            continue


        # ----------------------------------------------------
        # IMPORTANT:
        # Compare across ALL classes.
        #
        # This catches accidental cases where the same image
        # has been assigned different labels.
        # ----------------------------------------------------

        distance = (
            image_a["hash"]
            -
            image_b["hash"]
        )


        if (
            distance
            <= THRESHOLD
        ):

            duplicates.append({

                "distance":
                    distance,

                "class1":
                    image_a["class"],

                "split1":
                    image_a["split"],

                "image1":
                    image_a["path"],

                "class2":
                    image_b["class"],

                "split2":
                    image_b["split"],

                "image2":
                    image_b["path"]

            })


# ============================================================
# SORT RESULTS
# ============================================================

duplicates.sort(
    key=lambda x:
    x["distance"]
)


# ============================================================
# SHOW RESULTS
# ============================================================

print("=" * 70)

print(
    "CROSS-SPLIT NEAR-DUPLICATE RESULTS"
)

print("=" * 70)

print()

print(
    f"Found {len(duplicates)} "
    f"potential cross-split "
    f"near-duplicate pairs."
)

print()


# ============================================================
# SHOW DUPLICATES
# ============================================================

for item in duplicates:

    print(
        "-" * 70
    )

    print(
        f"Hash distance: "
        f"{item['distance']}"
    )

    print(
        f"Class 1:       "
        f"{item['class1']}"
    )

    print(
        f"Split 1:       "
        f"{item['split1']}"
    )

    print(
        f"Image 1:       "
        f"{item['image1']}"
    )

    print()

    print(
        f"Class 2:       "
        f"{item['class2']}"
    )

    print(
        f"Split 2:       "
        f"{item['split2']}"
    )

    print(
        f"Image 2:       "
        f"{item['image2']}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()

print(
    "=" * 70
)


if len(duplicates) == 0:

    print(
        "SUCCESS!"
    )

    print()

    print(
        "No cross-split near-duplicates "
        "were detected."
    )

    print()

    print(
        "The multiclass dataset is "
        "ready for training."
    )


else:

    print(
        "WARNING!"
    )

    print()

    print(
        "Potential cross-split "
        "near-duplicates were detected."
    )

    print()

    print(
        "Do NOT train yet."
    )

    print(
        "Review the duplicate pairs "
        "above first."
    )


print(
    "=" * 70
)