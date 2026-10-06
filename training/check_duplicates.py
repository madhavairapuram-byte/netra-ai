from pathlib import Path
from PIL import Image
import imagehash


BASE = Path("data/split")

FOLDERS = {
    "train": BASE / "train",
    "validation": BASE / "val",
    "test": BASE / "test"
}


# ------------------------------------------------
# Get all images
# ------------------------------------------------

images = []

for split_name, folder in FOLDERS.items():

    for class_folder in folder.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        for image_path in class_folder.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ]:
                continue

            images.append(
                (
                    split_name,
                    class_name,
                    image_path
                )
            )


print("=" * 70)
print("NEAR-DUPLICATE IMAGE CHECK")
print("=" * 70)

print(f"\nTotal images checked: {len(images)}")


# ------------------------------------------------
# Calculate perceptual hashes
# ------------------------------------------------

hashed_images = []

for index, (split, class_name, path) in enumerate(images):

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            hash_value = imagehash.phash(image)

            hashed_images.append(
                (
                    split,
                    class_name,
                    path,
                    hash_value
                )
            )

    except Exception as error:

        print(f"Could not process {path}: {error}")


# ------------------------------------------------
# Compare images from different dataset splits
# ------------------------------------------------

near_duplicates = []

for i in range(len(hashed_images)):

    split_a, class_a, path_a, hash_a = hashed_images[i]

    for j in range(i + 1, len(hashed_images)):

        split_b, class_b, path_b, hash_b = hashed_images[j]

        # We are mainly interested in leakage
        # between train/validation/test.

        if split_a == split_b:
            continue

        distance = hash_a - hash_b

        # Smaller distance = more visually similar
        if distance <= 6:

            near_duplicates.append(
                (
                    distance,
                    split_a,
                    class_a,
                    path_a,
                    split_b,
                    class_b,
                    path_b
                )
            )


# ------------------------------------------------
# Sort results
# ------------------------------------------------

near_duplicates.sort(
    key=lambda x: x[0]
)


# ------------------------------------------------
# Display results
# ------------------------------------------------

print()

if not near_duplicates:

    print("No strong near-duplicates were found between")
    print("training, validation and test sets.")

else:

    print(
        f"Found {len(near_duplicates)} "
        "potential near-duplicate pairs."
    )

    print()

    for result in near_duplicates[:50]:

        (
            distance,
            split_a,
            class_a,
            path_a,
            split_b,
            class_b,
            path_b
        ) = result

        print("-" * 70)

        print(f"Hash distance: {distance}")

        print(
            f"{split_a} / {class_a}:"
        )

        print(
            f"  {path_a}"
        )

        print(
            f"{split_b} / {class_b}:"
        )

        print(
            f"  {path_b}"
        )


print()
print("=" * 70)
print("NEAR-DUPLICATE CHECK COMPLETE")
print("=" * 70)