from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Studies\MSc Biomedical genetics 2025-2027\SEM 3\AI in healthcare\project\conjunctivitis_ai_app"
)

NEW_HEALTHY_DIR = Path(
    r"D:\Studies\MSc Biomedical genetics 2025-2027\SEM 3\AI in healthcare\project\Eye Conjunctiva Segmentation Dataset\Images"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ORIGINAL_NORMAL_DIR = RAW_DIR / "normal"
ORIGINAL_CONJUNCTIVITIS_DIR = RAW_DIR / "conjunctivitis"


# ============================================================
# SETTINGS
# ============================================================

# Same pHash threshold we used for the clean V4 split.
PHASH_THRESHOLD = 6

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(folder):
    """Return all supported image files inside a folder."""
    if not folder.exists():
        print(f"WARNING: Folder does not exist:")
        print(folder)
        return []

    return sorted(
        [
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def calculate_phash(path):
    """Calculate perceptual hash for one image."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            return imagehash.phash(image)
    except Exception as error:
        print(f"Could not read: {path}")
        print(f"Error: {error}")
        return None


def compare_hash_lists(reference_hashes, new_hashes, threshold):
    """
    Compare every new image against a reference collection.

    Returns:
        matches:
            list of tuples:
            (new_image, reference_image, distance)
    """

    matches = []

    for new_path, new_hash in new_hashes.items():

        if new_hash is None:
            continue

        best_match = None
        best_distance = None

        for reference_path, reference_hash in reference_hashes.items():

            if reference_hash is None:
                continue

            distance = new_hash - reference_hash

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = reference_path

        if best_distance is not None and best_distance <= threshold:
            matches.append(
                (
                    new_path,
                    best_match,
                    best_distance,
                )
            )

    return matches


def find_internal_duplicates(image_hashes, threshold):
    """
    Find duplicate/near-duplicate pairs within one dataset.
    """

    paths = list(image_hashes.keys())
    matches = []

    for i in range(len(paths)):
        path_a = paths[i]
        hash_a = image_hashes[path_a]

        if hash_a is None:
            continue

        for j in range(i + 1, len(paths)):
            path_b = paths[j]
            hash_b = image_hashes[path_b]

            if hash_b is None:
                continue

            distance = hash_a - hash_b

            if distance <= threshold:
                matches.append(
                    (
                        path_a,
                        path_b,
                        distance,
                    )
                )

    return matches


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("NEW HEALTHY DATASET DUPLICATE CHECK")
print("=" * 70)

print()
print("New healthy folder:")
print(NEW_HEALTHY_DIR)

print()
print("Original dataset:")
print(RAW_DIR)

print()
print("pHash threshold:", PHASH_THRESHOLD)

# ------------------------------------------------------------
# Collect images
# ------------------------------------------------------------

new_healthy_images = get_images(NEW_HEALTHY_DIR)
original_normal_images = get_images(ORIGINAL_NORMAL_DIR)
original_conj_images = get_images(ORIGINAL_CONJUNCTIVITIS_DIR)

print()
print("IMAGE COUNTS")
print("-" * 70)
print("New healthy:", len(new_healthy_images))
print("Original normal:", len(original_normal_images))
print("Original conjunctivitis:", len(original_conj_images))


# ------------------------------------------------------------
# Calculate hashes
# ------------------------------------------------------------

print()
print("Calculating perceptual hashes...")
print("This may take a little while because the images are high resolution.")

new_healthy_hashes = {}
original_normal_hashes = {}
original_conj_hashes = {}


for index, path in enumerate(new_healthy_images, start=1):

    new_healthy_hashes[path] = calculate_phash(path)

    if index % 50 == 0 or index == len(new_healthy_images):
        print(
            f"New healthy: {index}/{len(new_healthy_images)}"
        )


for index, path in enumerate(original_normal_images, start=1):

    original_normal_hashes[path] = calculate_phash(path)

    if index % 50 == 0 or index == len(original_normal_images):
        print(
            f"Original normal: {index}/{len(original_normal_images)}"
        )


for index, path in enumerate(original_conj_images, start=1):

    original_conj_hashes[path] = calculate_phash(path)

    if index % 50 == 0 or index == len(original_conj_images):
        print(
            f"Original conjunctivitis: {index}/{len(original_conj_images)}"
        )


# ------------------------------------------------------------
# Internal duplicates
# ------------------------------------------------------------

print()
print("=" * 70)
print("CHECKING DUPLICATES INSIDE THE 547 NEW HEALTHY IMAGES")
print("=" * 70)

internal_matches = find_internal_duplicates(
    new_healthy_hashes,
    PHASH_THRESHOLD,
)

print()
print("Internal duplicate/near-duplicate pairs:", len(internal_matches))

if internal_matches:

    print()
    print("MATCHES")
    print("-" * 70)

    for path_a, path_b, distance in internal_matches:

        print()
        print(f"Distance: {distance}")
        print(f"A: {path_a}")
        print(f"B: {path_b}")

else:

    print("No internal duplicates found at this threshold.")


# ------------------------------------------------------------
# Compare with original normal
# ------------------------------------------------------------

print()
print("=" * 70)
print("NEW HEALTHY vs ORIGINAL NORMAL")
print("=" * 70)

normal_matches = compare_hash_lists(
    original_normal_hashes,
    new_healthy_hashes,
    PHASH_THRESHOLD,
)

print()
print(
    "New healthy images with a possible match in original normal:",
    len(normal_matches),
)

if normal_matches:

    print()
    print("MATCHES")
    print("-" * 70)

    for new_path, original_path, distance in normal_matches:

        print()
        print(f"Distance: {distance}")
        print(f"New healthy: {new_path}")
        print(f"Original normal: {original_path}")

else:

    print("No matches found.")


# ------------------------------------------------------------
# Compare with original conjunctivitis
# ------------------------------------------------------------

print()
print("=" * 70)
print("NEW HEALTHY vs ORIGINAL CONJUNCTIVITIS")
print("=" * 70)

conj_matches = compare_hash_lists(
    original_conj_hashes,
    new_healthy_hashes,
    PHASH_THRESHOLD,
)

print()
print(
    "New healthy images with a possible match in original conjunctivitis:",
    len(conj_matches),
)

if conj_matches:

    print()
    print("MATCHES")
    print("-" * 70)

    for new_path, original_path, distance in conj_matches:

        print()
        print(f"Distance: {distance}")
        print(f"New healthy: {new_path}")
        print(f"Original conjunctivitis: {original_path}")

else:

    print("No matches found.")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()
print("New healthy images:", len(new_healthy_images))
print(
    "Internal duplicate/near-duplicate pairs:",
    len(internal_matches),
)
print(
    "Possible matches with original normal:",
    len(normal_matches),
)
print(
    "Possible matches with original conjunctivitis:",
    len(conj_matches),
)

print()
print("Nothing was deleted or modified.")
print("Duplicate detection is complete.")