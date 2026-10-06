from pathlib import Path
from PIL import Image
import imagehash


# ============================================================
# SETTINGS
# ============================================================

# Folder containing the 547 healthy-eye images
NORMAL_DIR = Path(
    r"D:\Studies\MSc Biomedical genetics 2025-2027\SEM 3\AI in healthcare\project\Eye Conjunctiva Segmentation Dataset\Images"
)

# Same threshold used in the previous duplicate check
PHASH_THRESHOLD = 6

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(folder):
    """Get all supported image files directly inside the folder."""

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def calculate_phash(path):
    """Calculate perceptual hash for an image."""

    try:
        with Image.open(path) as image:
            return imagehash.phash(
                image.convert("RGB")
            )

    except Exception as error:
        print()
        print(f"Could not read image:")
        print(path)
        print(f"Error: {error}")

        return None


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("KEEP ONE IMAGE PER HEALTHY-EYE SIMILARITY GROUP")
print("=" * 70)

print()
print("Target folder:")
print(NORMAL_DIR)

# ------------------------------------------------------------
# Check folder
# ------------------------------------------------------------

if not NORMAL_DIR.exists():

    print()
    print("ERROR: The target folder does not exist.")
    print(NORMAL_DIR)

    raise SystemExit


# ------------------------------------------------------------
# Get images
# ------------------------------------------------------------

images = get_images(NORMAL_DIR)

print()
print("Images currently in folder:", len(images))


# ------------------------------------------------------------
# SAFETY CHECK 1
# ------------------------------------------------------------

if len(images) != 547:

    print()
    print("=" * 70)
    print("SAFETY STOP")
    print("=" * 70)

    print()
    print("Expected exactly 547 images.")
    print(f"Found {len(images)} images instead.")

    print()
    print("NO FILES WILL BE DELETED.")

    raise SystemExit


# ------------------------------------------------------------
# Calculate pHash
# ------------------------------------------------------------

print()
print("Calculating perceptual hashes...")
print("Please wait.")

hashes = {}

for index, path in enumerate(images, start=1):

    hashes[path] = calculate_phash(path)

    if index % 50 == 0 or index == len(images):

        print(
            f"Processed {index}/{len(images)}"
        )


# ------------------------------------------------------------
# Check for failed images
# ------------------------------------------------------------

failed_images = [
    path
    for path in images
    if hashes[path] is None
]

if failed_images:

    print()
    print("=" * 70)
    print("SAFETY STOP")
    print("=" * 70)

    print()
    print(
        f"{len(failed_images)} image(s) could not be processed."
    )

    print()
    print("NO FILES WILL BE DELETED.")

    for path in failed_images:
        print(path)

    raise SystemExit


# ------------------------------------------------------------
# Build similarity groups
# ------------------------------------------------------------

print()
print("Grouping similar images...")
print("This may take some time.")

uf = UnionFind(images)

match_count = 0

for i in range(len(images)):

    path_a = images[i]
    hash_a = hashes[path_a]

    for j in range(i + 1, len(images)):

        path_b = images[j]
        hash_b = hashes[path_b]

        distance = hash_a - hash_b

        if distance <= PHASH_THRESHOLD:

            uf.union(
                path_a,
                path_b
            )

            match_count += 1


# ------------------------------------------------------------
# Create groups
# ------------------------------------------------------------

groups = {}

for path in images:

    root = uf.find(path)

    if root not in groups:
        groups[root] = []

    groups[root].append(path)


groups = list(groups.values())


# ------------------------------------------------------------
# Display group information
# ------------------------------------------------------------

print()
print("=" * 70)
print("GROUPING RESULT")
print("=" * 70)

print()
print("Images:", len(images))
print("Similarity matches:", match_count)
print("Similarity groups:", len(groups))


# ------------------------------------------------------------
# SAFETY CHECK 2
# ------------------------------------------------------------

if len(groups) != 399:

    print()
    print("=" * 70)
    print("SAFETY STOP")
    print("=" * 70)

    print()
    print("Expected exactly 399 similarity groups.")
    print(f"Found {len(groups)} groups instead.")

    print()
    print("NO FILES WILL BE DELETED.")

    raise SystemExit


# ------------------------------------------------------------
# Select one representative per group
# ------------------------------------------------------------

keep = set()

for group in groups:

    # Sort alphabetically so the selection is deterministic.
    group_sorted = sorted(
        group,
        key=lambda path: path.name.lower()
    )

    representative = group_sorted[0]

    keep.add(representative)


# ------------------------------------------------------------
# Determine files to delete
# ------------------------------------------------------------

delete = [
    path
    for path in images
    if path not in keep
]


# ------------------------------------------------------------
# Final safety checks
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL SAFETY CHECK")
print("=" * 70)

print()
print("Images before:", len(images))
print("Similarity groups:", len(groups))
print("Images to KEEP:", len(keep))
print("Images to DELETE:", len(delete))


if len(keep) != 399:

    print()
    print("SAFETY STOP")
    print("Expected 399 images to keep.")
    print("NO FILES WILL BE DELETED.")

    raise SystemExit


if len(delete) != 148:

    print()
    print("SAFETY STOP")
    print("Expected 148 images to delete.")
    print("NO FILES WILL BE DELETED.")

    raise SystemExit


# ------------------------------------------------------------
# Show deletion list before deleting
# ------------------------------------------------------------

print()
print("The following 148 images will be deleted:")

for path in delete:
    print(path.name)


# ------------------------------------------------------------
# DELETE
# ------------------------------------------------------------

print()
print("=" * 70)
print("DELETING NEAR-DUPLICATE IMAGES")
print("=" * 70)

deleted = 0

for path in delete:

    try:

        path.unlink()

        deleted += 1

    except Exception as error:

        print()
        print("Could not delete:")
        print(path)

        print("Error:")
        print(error)


# ------------------------------------------------------------
# Final count
# ------------------------------------------------------------

remaining = get_images(NORMAL_DIR)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print()
print("Successfully deleted:", deleted)
print("Images remaining:", len(remaining))


# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

if len(remaining) == 399:

    print()
    print("SUCCESS!")
    print("Exactly 399 healthy-eye images remain.")

else:

    print()
    print("WARNING!")
    print(
        f"Expected 399 images, but found {len(remaining)}."
    )

print()
print("The masks were not touched.")
print("Only images inside the Images folder were processed.")