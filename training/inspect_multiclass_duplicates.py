from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent

OUTPUT = (
    ROOT
    / "reports"
    / "multiclass_duplicate_review.jpg"
)


PAIRS = [

    (
        ROOT
        / "data"
        / "clean_multiclass"
        / "train"
        / "conjunctivitis"
        / "107 (2).jpg",

        ROOT
        / "data"
        / "clean_multiclass"
        / "test"
        / "Uveitis"
        / "189.jpeg"
    ),

    (
        ROOT
        / "data"
        / "clean_multiclass"
        / "train"
        / "conjunctivitis"
        / "112 (2).jpg",

        ROOT
        / "data"
        / "clean_multiclass"
        / "test"
        / "Uveitis"
        / "189.jpeg"
    ),

    (
        ROOT
        / "data"
        / "clean_multiclass"
        / "train"
        / "conjunctivitis"
        / "158.jpeg",

        ROOT
        / "data"
        / "clean_multiclass"
        / "test"
        / "Uveitis"
        / "189.jpeg"
    ),

    (
        ROOT
        / "data"
        / "clean_multiclass"
        / "train"
        / "stye"
        / "326.jpeg",

        ROOT
        / "data"
        / "clean_multiclass"
        / "val"
        / "conjunctivitis"
        / "680.png"
    ),

    (
        ROOT
        / "data"
        / "clean_multiclass"
        / "train"
        / "Uveitis"
        / "13.jpeg",

        ROOT
        / "data"
        / "clean_multiclass"
        / "val"
        / "conjunctivitis"
        / "aug_832.jpg"
    )
]


# ============================================================
# SETTINGS
# ============================================================

IMAGE_WIDTH = 500

IMAGE_HEIGHT = 400

LABEL_HEIGHT = 80

PAIR_HEIGHT = (
    IMAGE_HEIGHT
    + LABEL_HEIGHT
)


# ============================================================
# CREATE CANVAS
# ============================================================

canvas_width = (
    IMAGE_WIDTH * 2
)

canvas_height = (
    PAIR_HEIGHT
    * len(PAIRS)
)


canvas = Image.new(
    "RGB",
    (
        canvas_width,
        canvas_height
    ),
    "white"
)


draw = ImageDraw.Draw(
    canvas
)


# ============================================================
# LOAD AND DRAW PAIRS
# ============================================================

for index, pair in enumerate(
    PAIRS
):

    left_path = pair[0]

    right_path = pair[1]


    y = (
        index
        * PAIR_HEIGHT
    )


    # --------------------------------------------------------
    # LEFT IMAGE
    # --------------------------------------------------------

    left_image = Image.open(
        left_path
    ).convert(
        "RGB"
    )

    left_image.thumbnail(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
    )


    left_x = (
        IMAGE_WIDTH
        - left_image.width
    ) // 2


    left_y = (
        y
        + (
            IMAGE_HEIGHT
            - left_image.height
        ) // 2
    )


    canvas.paste(
        left_image,
        (
            left_x,
            left_y
        )
    )


    # --------------------------------------------------------
    # RIGHT IMAGE
    # --------------------------------------------------------

    right_image = Image.open(
        right_path
    ).convert(
        "RGB"
    )

    right_image.thumbnail(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
    )


    right_x = (
        IMAGE_WIDTH
        + (
            IMAGE_WIDTH
            - right_image.width
        ) // 2
    )


    right_y = (
        y
        + (
            IMAGE_HEIGHT
            - right_image.height
        ) // 2
    )


    canvas.paste(
        right_image,
        (
            right_x,
            right_y
        )
    )


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    left_label = (
        f"LEFT: "
        f"{left_path.parent.name}/"
        f"{left_path.name}"
    )

    right_label = (
        f"RIGHT: "
        f"{right_path.parent.name}/"
        f"{right_path.name}"
    )


    draw.text(
        (
            10,
            y
            + IMAGE_HEIGHT
            + 10
        ),
        left_label,
        fill="black"
    )


    draw.text(
        (
            IMAGE_WIDTH + 10,
            y
            + IMAGE_HEIGHT
            + 10
        ),
        right_label,
        fill="black"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


canvas.save(
    OUTPUT,
    quality=95
)


print("=" * 70)

print(
    "DUPLICATE REVIEW IMAGE CREATED"
)

print("=" * 70)

print()

print(
    "Saved to:"
)

print(
    OUTPUT
)

print()

print(
    "Open this image and inspect all five pairs."
)

print(
    "Nothing in the dataset was modified."
)

print("=" * 70)