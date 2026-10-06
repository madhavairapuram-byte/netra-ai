import os
import csv
import math

from PIL import Image, ImageStat


# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "data/clean_split/train"

EXTERNAL_NORMAL_DIR = "data/external/healthy_eye"
EXTERNAL_CONJUNCTIVITIS_DIR = "data/external/infected_eye"

OUTPUT_CSV = "reports/dataset_shift_analysis.csv"

EXCLUDED_FILE = "170.jpeg"

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    paths = []

    if not os.path.exists(folder):

        print()
        print("ERROR: Folder does not exist:")
        print(folder)
        return paths

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
            continue

        if not filename.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        if filename.lower() == EXCLUDED_FILE.lower():
            continue

        paths.append(path)

    return sorted(paths)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            width, height = image.size

            aspect_ratio = (
                width / height
                if height != 0
                else 0
            )

            stat = ImageStat.Stat(image)

            red_mean = stat.mean[0]
            green_mean = stat.mean[1]
            blue_mean = stat.mean[2]

            overall_brightness = (
                red_mean
                + green_mean
                + blue_mean
            ) / 3.0

            red_green_difference = (
                red_mean - green_mean
            )

            red_blue_difference = (
                red_mean - blue_mean
            )

            # Approximate colour saturation.
            # This is not HSV saturation, but gives a useful
            # simple measure of RGB channel spread.

            saturation_proxy = (
                max(
                    red_mean,
                    green_mean,
                    blue_mean
                )
                -
                min(
                    red_mean,
                    green_mean,
                    blue_mean
                )
            )

            return {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "red_mean": red_mean,
                "green_mean": green_mean,
                "blue_mean": blue_mean,
                "brightness": overall_brightness,
                "red_green_difference":
                    red_green_difference,
                "red_blue_difference":
                    red_blue_difference,
                "saturation_proxy":
                    saturation_proxy,
                "error": ""
            }

    except Exception as error:

        return {
            "width": 0,
            "height": 0,
            "aspect_ratio": 0,
            "red_mean": 0,
            "green_mean": 0,
            "blue_mean": 0,
            "brightness": 0,
            "red_green_difference": 0,
            "red_blue_difference": 0,
            "saturation_proxy": 0,
            "error": str(error)
        }


# ============================================================
# COLLECT DATA
# ============================================================

print("=" * 70)
print("DATASET SHIFT ANALYSIS")
print("=" * 70)

print()

print("This script does NOT train a model.")
print("It only compares image characteristics.")
print()


# ------------------------------------------------------------
# Training dataset
# ------------------------------------------------------------

training_normal = get_images(
    os.path.join(
        TRAIN_DIR,
        "normal"
    )
)

training_conjunctivitis = get_images(
    os.path.join(
        TRAIN_DIR,
        "conjunctivitis"
    )
)


# ------------------------------------------------------------
# External dataset
# ------------------------------------------------------------

external_normal = get_images(
    EXTERNAL_NORMAL_DIR
)

external_conjunctivitis = get_images(
    EXTERNAL_CONJUNCTIVITIS_DIR
)


# ============================================================
# SUMMARY
# ============================================================

print("-" * 70)
print("DATASET COUNTS")
print("-" * 70)

print()

print(
    "Training normal:",
    len(training_normal)
)

print(
    "Training conjunctivitis:",
    len(training_conjunctivitis)
)

print()

print(
    "External normal:",
    len(external_normal)
)

print(
    "External conjunctivitis:",
    len(external_conjunctivitis)
)

print()

all_images = []


def add_images(
    paths,
    dataset,
    label
):

    for path in paths:

        all_images.append(
            {
                "dataset": dataset,
                "label": label,
                "filename":
                    os.path.basename(path),
                "path": path
            }
        )


add_images(
    training_normal,
    "training",
    "normal"
)

add_images(
    training_conjunctivitis,
    "training",
    "conjunctivitis"
)

add_images(
    external_normal,
    "external",
    "normal"
)

add_images(
    external_conjunctivitis,
    "external",
    "conjunctivitis"
)


# ============================================================
# ANALYZE
# ============================================================

print("-" * 70)
print("ANALYZING IMAGE CHARACTERISTICS")
print("-" * 70)

print()

rows = []

total = len(all_images)

for index, item in enumerate(
    all_images,
    start=1
):

    features = analyze_image(
        item["path"]
    )

    row = {

        "dataset":
            item["dataset"],

        "label":
            item["label"],

        "filename":
            item["filename"],

        "width":
            features["width"],

        "height":
            features["height"],

        "aspect_ratio":
            features["aspect_ratio"],

        "brightness":
            features["brightness"],

        "red_mean":
            features["red_mean"],

        "green_mean":
            features["green_mean"],

        "blue_mean":
            features["blue_mean"],

        "red_green_difference":
            features[
                "red_green_difference"
            ],

        "red_blue_difference":
            features[
                "red_blue_difference"
            ],

        "saturation_proxy":
            features[
                "saturation_proxy"
            ],

        "error":
            features["error"]
    }

    rows.append(row)

    if (
        index % 100 == 0
        or index == total
    ):

        print(
            f"Analyzed "
            f"{index}/{total}"
        )


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    "reports",
    exist_ok=True
)

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "dataset",
        "label",
        "filename",
        "width",
        "height",
        "aspect_ratio",
        "brightness",
        "red_mean",
        "green_mean",
        "blue_mean",
        "red_green_difference",
        "red_blue_difference",
        "saturation_proxy",
        "error"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def mean(values):

    values = [
        value
        for value in values
        if isinstance(value, (int, float))
        and not math.isnan(value)
    ]

    if len(values) == 0:
        return 0.0

    return sum(values) / len(values)


def get_rows(
    dataset=None,
    label=None
):

    selected = rows

    if dataset is not None:

        selected = [
            row
            for row in selected
            if row["dataset"] == dataset
        ]

    if label is not None:

        selected = [
            row
            for row in selected
            if row["label"] == label
        ]

    return selected


def print_group(
    title,
    group
):

    print()
    print(title)
    print("-" * len(title))

    print(
        "Images:",
        len(group)
    )

    if len(group) == 0:
        return

    print(
        "Mean width:",
        f"{mean([r['width'] for r in group]):.1f}"
    )

    print(
        "Mean height:",
        f"{mean([r['height'] for r in group]):.1f}"
    )

    print(
        "Mean aspect ratio:",
        f"{mean([r['aspect_ratio'] for r in group]):.3f}"
    )

    print(
        "Mean brightness:",
        f"{mean([r['brightness'] for r in group]):.2f}"
    )

    print(
        "Mean red:",
        f"{mean([r['red_mean'] for r in group]):.2f}"
    )

    print(
        "Mean green:",
        f"{mean([r['green_mean'] for r in group]):.2f}"
    )

    print(
        "Mean blue:",
        f"{mean([r['blue_mean'] for r in group]):.2f}"
    )

    print(
        "Mean red-green difference:",
        f"{mean([r['red_green_difference'] for r in group]):.2f}"
    )

    print(
        "Mean red-blue difference:",
        f"{mean([r['red_blue_difference'] for r in group]):.2f}"
    )

    print(
        "Mean saturation proxy:",
        f"{mean([r['saturation_proxy'] for r in group]):.2f}"
    )


# ============================================================
# PRINT GROUPS
# ============================================================

training_all = get_rows(
    dataset="training"
)

external_all = get_rows(
    dataset="external"
)

training_normal_rows = get_rows(
    dataset="training",
    label="normal"
)

training_conj_rows = get_rows(
    dataset="training",
    label="conjunctivitis"
)

external_normal_rows = get_rows(
    dataset="external",
    label="normal"
)

external_conj_rows = get_rows(
    dataset="external",
    label="conjunctivitis"
)


print()
print("=" * 70)
print("IMAGE CHARACTERISTICS")
print("=" * 70)

print_group(
    "TRAINING DATASET - ALL",
    training_all
)

print_group(
    "TRAINING DATASET - NORMAL",
    training_normal_rows
)

print_group(
    "TRAINING DATASET - CONJUNCTIVITIS",
    training_conj_rows
)

print_group(
    "EXTERNAL DATASET - ALL",
    external_all
)

print_group(
    "EXTERNAL DATASET - NORMAL",
    external_normal_rows
)

print_group(
    "EXTERNAL DATASET - CONJUNCTIVITIS",
    external_conj_rows
)


# ============================================================
# DIRECT SOURCE COMPARISON
# ============================================================

print()
print("=" * 70)
print("TRAINING VS EXTERNAL DATASET")
print("=" * 70)

print()

training_brightness = mean(
    [
        r["brightness"]
        for r in training_all
    ]
)

external_brightness = mean(
    [
        r["brightness"]
        for r in external_all
    ]
)

training_aspect = mean(
    [
        r["aspect_ratio"]
        for r in training_all
    ]
)

external_aspect = mean(
    [
        r["aspect_ratio"]
        for r in external_all
    ]
)

training_red = mean(
    [
        r["red_mean"]
        for r in training_all
    ]
)

external_red = mean(
    [
        r["red_mean"]
        for r in external_all
    ]
)

training_green = mean(
    [
        r["green_mean"]
        for r in training_all
    ]
)

external_green = mean(
    [
        r["green_mean"]
        for r in external_all
    ]
)

training_blue = mean(
    [
        r["blue_mean"]
        for r in training_all
    ]
)

external_blue = mean(
    [
        r["blue_mean"]
        for r in external_all
    ]
)

training_saturation = mean(
    [
        r["saturation_proxy"]
        for r in training_all
    ]
)

external_saturation = mean(
    [
        r["saturation_proxy"]
        for r in external_all
    ]
)


print(
    f"Brightness:"
)

print(
    f"  Training: {training_brightness:.2f}"
)

print(
    f"  External: {external_brightness:.2f}"
)

print()

print(
    f"Aspect ratio:"
)

print(
    f"  Training: {training_aspect:.3f}"
)

print(
    f"  External: {external_aspect:.3f}"
)

print()

print(
    f"Red channel:"
)

print(
    f"  Training: {training_red:.2f}"
)

print(
    f"  External: {external_red:.2f}"
)

print()

print(
    f"Green channel:"
)

print(
    f"  Training: {training_green:.2f}"
)

print(
    f"  External: {external_green:.2f}"
)

print()

print(
    f"Blue channel:"
)

print(
    f"  Training: {training_blue:.2f}"
)

print(
    f"  External: {external_blue:.2f}"
)

print()

print(
    f"Saturation proxy:"
)

print(
    f"  Training: {training_saturation:.2f}"
)

print(
    f"  External: {external_saturation:.2f}"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("DATASET SHIFT ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    "Detailed results saved to:"
)

print(
    OUTPUT_CSV
)

print()

print(
    "No images were modified."
)

print(
    "No model was trained."
)

print("=" * 70)