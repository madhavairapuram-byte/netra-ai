from pathlib import Path
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parent.parent

folders = {
    "normal": ROOT / "data" / "raw" / "normal",
    "conjunctivitis": ROOT / "data" / "raw" / "conjunctivitis",
}

# Hash distance at or below this value will be reported
THRESHOLD = 6

images = []

print("=" * 70)
print("CHECKING ORIGINAL RAW DATASET FOR NEAR-DUPLICATES")
print("=" * 70)

# ---------------------------------------------------------
# Calculate perceptual hashes for all original images
# ---------------------------------------------------------

for label, folder in folders.items():

    files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    print(f"{label}: {len(files)} images")

    for file in files:

        try:
            img = Image.open(file).convert("RGB")
            hash_value = imagehash.phash(img)

            images.append({
                "label": label,
                "path": file,
                "hash": hash_value
            })

        except Exception as e:
            print(f"Could not read {file}: {e}")

print()
print(f"Total images checked: {len(images)}")
print()

# ---------------------------------------------------------
# Compare every image with every other image
# ---------------------------------------------------------

duplicates = []

for i in range(len(images)):

    for j in range(i + 1, len(images)):

        # Only compare images from the same disease category
        if images[i]["label"] != images[j]["label"]:
            continue

        distance = images[i]["hash"] - images[j]["hash"]

        if distance <= THRESHOLD:

            duplicates.append({
                "distance": distance,
                "label": images[i]["label"],
                "image1": images[i]["path"],
                "image2": images[j]["path"],
            })

# ---------------------------------------------------------
# Sort by most similar first
# ---------------------------------------------------------

duplicates.sort(key=lambda x: x["distance"])

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Found {len(duplicates)} potential near-duplicate pairs.")
print()

for item in duplicates:

    print("-" * 70)

    print(f"Hash distance: {item['distance']}")
    print(f"Category:      {item['label']}")

    print(f"Image 1:")
    print(f"  {item['image1']}")

    print(f"Image 2:")
    print(f"  {item['image2']}")

print()
print("=" * 70)
print("RAW DATASET CHECK COMPLETE")
print("=" * 70)