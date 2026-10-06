from pathlib import Path
from PIL import Image
import imagehash


# ============================================================
# PATH
# ============================================================

NEW_HEALTHY_DIR = Path(
    r"D:\Studies\MSc Biomedical genetics 2025-2027\SEM 3\AI in healthcare\project\Eye Conjunctiva Segmentation Dataset\Images"
)

PHASH_THRESHOLD = 6

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# LOAD IMAGES
# ============================================================

def get_images(folder):
    return sorted(
        [
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def calculate_phash(path):
    try:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("RGB"))
    except Exception as error:
        print(f"Could not read: {path}")
        print(error)
        return None


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
# MAIN
# ============================================================

print("=" * 70)
print("GROUPING NEW HEALTHY IMAGES BY PERCEPTUAL SIMILARITY")
print("=" * 70)

images = get_images(NEW_HEALTHY_DIR)

print()
print("Images found:", len(images))
print("pHash threshold:", PHASH_THRESHOLD)

# ------------------------------------------------------------
# Calculate hashes
# ------------------------------------------------------------

print()
print("Calculating pHash values...")

hashes = {}

for index, path in enumerate(images, start=1):

    hashes[path] = calculate_phash(path)

    if index % 50 == 0 or index == len(images):
        print(f"Processed {index}/{len(images)}")


# ------------------------------------------------------------
# Build similarity groups
# ------------------------------------------------------------

print()
print("Comparing images...")

valid_images = [
    path for path in images
    if hashes[path] is not None
]

uf = UnionFind(valid_images)

comparison_count = 0
match_count = 0

for i in range(len(valid_images)):

    path_a = valid_images[i]
    hash_a = hashes[path_a]

    for j in range(i + 1, len(valid_images)):

        path_b = valid_images[j]
        hash_b = hashes[path_b]

        distance = hash_a - hash_b

        comparison_count += 1

        if distance <= PHASH_THRESHOLD:
            uf.union(path_a, path_b)
            match_count += 1


# ------------------------------------------------------------
# Build groups
# ------------------------------------------------------------

groups = {}

for path in valid_images:

    root = uf.find(path)

    if root not in groups:
        groups[root] = []

    groups[root].append(path)


groups = sorted(
    groups.values(),
    key=lambda group: (-len(group), str(group[0]))
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print()
print("Total images:", len(images))
print("Valid images:", len(valid_images))
print("Similarity matches:", match_count)
print("Independent similarity groups:", len(groups))

print()
print(
    "Images that could be retained as one representative per group:",
    len(groups),
)

print()
print("Images that are part of groups containing >1 image:")

multi_image_groups = [
    group for group in groups
    if len(group) > 1
]

print("Number of multi-image groups:", len(multi_image_groups))


# ------------------------------------------------------------
# Show groups
# ------------------------------------------------------------

print()
print("=" * 70)
print("NEAR-DUPLICATE GROUPS")
print("=" * 70)

for number, group in enumerate(multi_image_groups, start=1):

    print()
    print(f"GROUP {number} ({len(group)} images)")
    print("-" * 70)

    for path in group:
        print(path)


# ------------------------------------------------------------
# Group size distribution
# ------------------------------------------------------------

from collections import Counter

group_sizes = Counter(len(group) for group in groups)

print()
print("=" * 70)
print("GROUP SIZE DISTRIBUTION")
print("=" * 70)

for size in sorted(group_sizes):
    print(
        f"Groups containing {size} image(s):",
        group_sizes[size]
    )


print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print("No images were deleted.")
print("No images were moved.")
print("No images were renamed.")
print()
print("This script only identified similarity groups.")