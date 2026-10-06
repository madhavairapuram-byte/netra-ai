from PIL import Image, ImageOps, ImageDraw
from pathlib import Path

# ---------------------------------------------------------
# List of suspicious near-duplicate image pairs
# ---------------------------------------------------------

pairs = [
    (0, "data/split/train/conjunctivitis/aug_802.jpg",
        "data/split/val/conjunctivitis/679.png"),

    (2, "data/split/train/conjunctivitis/325.jpg",
        "data/split/val/conjunctivitis/324.jpg"),

    (4, "data/split/train/normal/236.jpg",
        "data/split/val/normal/357.jpg"),

    (6, "data/split/train/conjunctivitis/104.jpg",
        "data/split/val/conjunctivitis/533.jpg"),

    (6, "data/split/train/conjunctivitis/190.jpg",
        "data/split/test/conjunctivitis/202.jpg"),

    (6, "data/split/train/conjunctivitis/195.jpg",
        "data/split/val/conjunctivitis/206.jpg"),

    (6, "data/split/train/conjunctivitis/292.jpg",
        "data/split/test/conjunctivitis/378.jpg"),

    (6, "data/split/train/conjunctivitis/360.jpg",
        "data/split/test/conjunctivitis/199.jpg"),

    (6, "data/split/train/conjunctivitis/365.jpg",
        "data/split/val/conjunctivitis/280.jpg"),

    (6, "data/split/train/normal/244.jpg",
        "data/split/val/normal/698.jpg"),

    (6, "data/split/train/normal/281.jpg",
        "data/split/test/normal/15.jpg"),

    (6, "data/split/train/normal/506.jpg",
        "data/split/test/normal/341.jpg"),

    (6, "data/split/train/normal/532.jpg",
        "data/split/test/normal/34.jpg"),

    (6, "data/split/train/normal/545.jpg",
        "data/split/test/normal/648.jpg"),

    (6, "data/split/train/normal/74.jpg",
        "data/split/val/normal/564.jpg"),

    (6, "data/split/val/normal/698.jpg",
        "data/split/test/normal/474.jpg"),
]

# ---------------------------------------------------------
# Project location
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Where the final report will be saved
OUTPUT = ROOT / "reports" / "near_duplicate_report.jpg"

# ---------------------------------------------------------
# Image/report settings
# ---------------------------------------------------------

thumb_w = 350
thumb_h = 260

row_h = 330
label_h = 45

# Create a large white report image
report = Image.new(
    "RGB",
    (thumb_w * 2 + 40, row_h * len(pairs)),
    "white"
)

draw = ImageDraw.Draw(report)

# ---------------------------------------------------------
# Add every suspicious pair to the report
# ---------------------------------------------------------

for i, (distance, path1, path2) in enumerate(pairs):

    y = i * row_h

    # Open both images
    img1 = Image.open(ROOT / path1).convert("RGB")
    img2 = Image.open(ROOT / path2).convert("RGB")

    # Resize images while keeping their original proportions
    img1 = ImageOps.contain(
        img1,
        (thumb_w, thumb_h)
    )

    img2 = ImageOps.contain(
        img2,
        (thumb_w, thumb_h)
    )

    # Positions of the two images
    x1 = 10
    x2 = thumb_w + 25

    # Put images into the report
    report.paste(
        img1,
        (x1, y + label_h)
    )

    report.paste(
        img2,
        (x2, y + label_h)
    )

    # Write labels
    draw.text(
        (x1, y + 5),
        f"Pair {i + 1} | Hash distance: {distance}",
        fill="black"
    )

    draw.text(
        (x2, y + 5),
        f"Pair {i + 1}",
        fill="black"
    )

# ---------------------------------------------------------
# Create reports folder if it doesn't exist
# ---------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Save the final report
report.save(
    OUTPUT,
    quality=95
)

# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print("=" * 70)
print("DUPLICATE VISUAL REPORT CREATED")
print("=" * 70)

print("Saved to:")
print(OUTPUT)

print("=" * 70)