from pathlib import Path
from PIL import Image
import hashlib
import imagehash
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "external_dataset_duplicate_report.csv"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# Same threshold we used for the original dataset.
# <= 6 = potential near-duplicate
PHASH_THRESHOLD = 6


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(folder):
    """
    Find all image files inside a folder, including subfolders.
    """
    images = []

    if not folder.exists():
        return images

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def md5_hash(path):
    """
    Calculate exact MD5 hash of a file.
    """
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def perceptual_hash(path):
    """
    Calculate perceptual hash of an image.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img)
    except Exception as e:
        print(f"Could not process image: {path}")
        print(f"Error: {e}")
        return None


# ============================================================
# CHECK FOLDER STRUCTURE
# ============================================================

healthy_dir = EXTERNAL_DIR / "healthy_eye"
infected_dir = EXTERNAL_DIR / "infected_eye"

print("=" * 70)
print("EXTERNAL DATASET CHECK")
print("=" * 70)

print()
print(f"Project folder:")
print(PROJECT_ROOT)

print()
print(f"External dataset:")
print(EXTERNAL_DIR)

print()
print("Checking folders...")

if not healthy_dir.exists():
    print()
    print("ERROR: healthy_eye folder was not found.")
    print(f"Expected location:")
    print(healthy_dir)
    raise SystemExit(1)

if not infected_dir.exists():
    print()
    print("ERROR: infected_eye folder was not found.")
    print(f"Expected location:")
    print(infected_dir)
    raise SystemExit(1)


# ============================================================
# GET EXTERNAL IMAGES
# ============================================================

healthy_images = get_images(healthy_dir)
infected_images = get_images(infected_dir)

external_images = healthy_images + infected_images

print()
print("-" * 70)
print("EXTERNAL DATASET IMAGE COUNTS")
print("-" * 70)

print(f"Healthy images       : {len(healthy_images)}")
print(f"Infected images      : {len(infected_images)}")
print(f"Total images         : {len(external_images)}")

print()

# Expected counts from the published dataset description
expected_healthy = 181
expected_infected = 177
expected_total = 358

print("Expected according to published dataset description:")
print(f"Healthy              : {expected_healthy}")
print(f"Infected             : {expected_infected}")
print(f"Total                : {expected_total}")

print()

if (
    len(healthy_images) == expected_healthy
    and len(infected_images) == expected_infected
):
    print("COUNT CHECK: PASSED")
else:
    print("COUNT CHECK: DIFFERENT FROM REPORTED COUNTS")
    print("This is not necessarily a problem.")
    print("We will inspect the actual files before continuing.")


# ============================================================
# GET ORIGINAL RAW DATASET
# ============================================================

raw_normal_dir = RAW_DIR / "normal"
raw_conj_dir = RAW_DIR / "conjunctivitis"

raw_normal_images = get_images(raw_normal_dir)
raw_conj_images = get_images(raw_conj_dir)

raw_images = raw_normal_images + raw_conj_images

print()
print("-" * 70)
print("ORIGINAL DATASET COUNTS")
print("-" * 70)

print(f"Original normal images          : {len(raw_normal_images)}")
print(f"Original conjunctivitis images  : {len(raw_conj_images)}")
print(f"Original total                  : {len(raw_images)}")


# ============================================================
# EXACT DUPLICATE CHECK
# ============================================================

print()
print("-" * 70)
print("CHECKING EXACT DUPLICATES")
print("-" * 70)

print("Creating MD5 hashes...")

raw_md5 = {}

for path in raw_images:
    try:
        file_hash = md5_hash(path)
        raw_md5.setdefault(file_hash, []).append(path)
    except Exception as e:
        print(f"Could not hash {path}: {e}")


exact_matches = []

for external_path in external_images:

    try:
        file_hash = md5_hash(external_path)

        if file_hash in raw_md5:

            for raw_path in raw_md5[file_hash]:

                exact_matches.append(
                    (
                        external_path,
                        raw_path
                    )
                )

    except Exception as e:
        print(f"Could not hash {external_path}: {e}")


print()
print(f"Exact duplicate matches found: {len(exact_matches)}")


if len(exact_matches) > 0:

    print()
    print("EXACT DUPLICATES:")
    print()

    for external_path, raw_path in exact_matches:

        print(f"External : {external_path}")
        print(f"Original : {raw_path}")
        print("-" * 50)

else:

    print("No exact duplicates found.")


# ============================================================
# PERCEPTUAL HASH CHECK
# ============================================================

print()
print("-" * 70)
print("CHECKING NEAR-DUPLICATES")
print("-" * 70)

print()
print("Calculating perceptual hashes...")
print("This may take a little while.")

raw_phashes = []

for path in raw_images:

    phash = perceptual_hash(path)

    if phash is not None:
        raw_phashes.append(
            (path, phash)
        )


external_phashes = []

for path in external_images:

    phash = perceptual_hash(path)

    if phash is not None:
        external_phashes.append(
            (path, phash)
        )


print()
print(f"Original images hashed : {len(raw_phashes)}")
print(f"External images hashed : {len(external_phashes)}")


# ============================================================
# COMPARE HASHES
# ============================================================

near_duplicates = []

print()
print("Comparing external images against original images...")

for external_path, external_hash in external_phashes:

    for raw_path, raw_hash in raw_phashes:

        distance = external_hash - raw_hash

        if distance <= PHASH_THRESHOLD:

            near_duplicates.append(
                (
                    external_path,
                    raw_path,
                    distance
                )
            )


# Sort by smallest distance first
near_duplicates.sort(
    key=lambda x: x[2]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("-" * 70)
print("NEAR-DUPLICATE RESULTS")
print("-" * 70)

print()
print(f"Potential near-duplicate matches: {len(near_duplicates)}")

if len(near_duplicates) == 0:

    print()
    print("NO POTENTIAL NEAR-DUPLICATES FOUND.")
    print()
    print("This is a very good result.")

else:

    print()
    print("Potential matches:")
    print()

    for external_path, raw_path, distance in near_duplicates:

        print(f"Distance : {distance}")
        print(f"External : {external_path}")
        print(f"Original : {raw_path}")
        print("-" * 60)


# ============================================================
# SAVE CSV REPORT
# ============================================================

print()
print("-" * 70)
print("SAVING REPORT")
print("-" * 70)

with open(
    REPORT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow(
        [
            "external_image",
            "original_image",
            "phash_distance",
            "classification"
        ]
    )

    for external_path, raw_path, distance in near_duplicates:

        if distance == 0:
            classification = "EXACT_PERCEPTUAL_MATCH"

        elif distance <= 2:
            classification = "VERY_STRONG_MATCH"

        elif distance <= 4:
            classification = "STRONG_MATCH"

        else:
            classification = "POSSIBLE_MATCH"

        writer.writerow(
            [
                str(external_path),
                str(raw_path),
                distance,
                classification
            ]
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print(f"External healthy images      : {len(healthy_images)}")
print(f"External infected images     : {len(infected_images)}")
print(f"External total               : {len(external_images)}")

print()
print(f"Exact duplicate matches      : {len(exact_matches)}")
print(f"Near-duplicate matches       : {len(near_duplicates)}")

print()
print(f"Report saved to:")
print(REPORT_FILE)

print()

if len(exact_matches) == 0 and len(near_duplicates) == 0:

    print("RESULT: EXTERNAL DATASET LOOKS INDEPENDENT")
    print()
    print("We can proceed to external validation.")

elif len(exact_matches) == 0:

    print("RESULT: NO EXACT DUPLICATES, BUT SOME NEAR-DUPLICATES EXIST.")
    print()
    print("We should inspect the near-duplicate report before testing.")

else:

    print("RESULT: EXACT DUPLICATES EXIST.")
    print()
    print("We should investigate them before calling this an external test.")

print()
print("=" * 70)