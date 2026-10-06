from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont

from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "conjunctivitis_efficientnet_b0_clean.pth"
)

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_split"
    / "train"
)

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "gradcam_external"
)


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

# Known duplicate that we already identified.
EXCLUDED_IMAGE = "170.jpeg"

# Number of false-positive healthy images to visualize.
NUMBER_TO_SHOW = 20


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("GRAD-CAM ANALYSIS")
print("=" * 70)

print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    print()
    print("ERROR: Model file not found:")
    print(MODEL_PATH)

    raise SystemExit(1)


if not TRAIN_DIR.exists():

    print()
    print("ERROR: Training directory not found:")
    print(TRAIN_DIR)

    raise SystemExit(1)


if not EXTERNAL_DIR.exists():

    print()
    print("ERROR: External dataset not found:")
    print(EXTERNAL_DIR)

    raise SystemExit(1)


# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = sorted(
    [
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ]
)


if len(class_names) != 2:

    print()
    print("ERROR: Expected two classes.")

    print(
        f"Classes found: {class_names}"
    )

    raise SystemExit(1)


class_to_idx = {
    name: index
    for index, name in enumerate(class_names)
}


normal_class = None
conjunctivitis_class = None


for name in class_names:

    if "normal" in name.lower():

        normal_class = name

    if "conjunctivitis" in name.lower():

        conjunctivitis_class = name


if normal_class is None:

    print("ERROR: Normal class not found.")
    raise SystemExit(1)


if conjunctivitis_class is None:

    print("ERROR: Conjunctivitis class not found.")
    raise SystemExit(1)


NORMAL_INDEX = class_to_idx[normal_class]

CONJUNCTIVITIS_INDEX = class_to_idx[
    conjunctivitis_class
]


print()
print("Class mapping:")

for index, name in enumerate(class_names):

    print(
        f"  {index} = {name}"
    )


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

FALSE_POSITIVE_DIR = (
    OUTPUT_DIR
    / "false_positive_healthy_predicted_conjunctivitis"
)

FALSE_POSITIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("-" * 70)
print("LOADING MODEL")
print("-" * 70)


model = models.efficientnet_b0(
    weights=None
)


model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict
)


model = model.to(
    DEVICE
)


model.eval()


print()
print("Model loaded successfully.")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# FIND EXTERNAL IMAGES
# ============================================================

healthy_dir = (
    EXTERNAL_DIR
    / "healthy_eye"
)

infected_dir = (
    EXTERNAL_DIR
    / "infected_eye"
)


def get_images(folder):

    images = []

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.name.lower() == EXCLUDED_IMAGE.lower():
            continue

        images.append(path)

    return sorted(images)


healthy_images = get_images(
    healthy_dir
)

infected_images = get_images(
    infected_dir
)


print()
print("-" * 70)
print("EXTERNAL DATASET")
print("-" * 70)

print()
print(
    f"Healthy images: {len(healthy_images)}"
)

print(
    f"Conjunctivitis images: {len(infected_images)}"
)


# ============================================================
# STEP 1
# FIND FALSE POSITIVE HEALTHY IMAGES
# ============================================================

print()
print("-" * 70)
print("FINDING FALSE POSITIVES")
print("-" * 70)


false_positives = []


with torch.no_grad():

    for counter, image_path in enumerate(
        healthy_images,
        start=1
    ):

        image = Image.open(
            image_path
        ).convert("RGB")


        tensor = transform(
            image
        )


        tensor = (
            tensor
            .unsqueeze(0)
            .to(DEVICE)
        )


        output = model(
            tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()


        conjunctivitis_probability = (
            probabilities[
                0,
                CONJUNCTIVITIS_INDEX
            ].item()
        )


        # Healthy image incorrectly classified
        # as conjunctivitis.
        if predicted_index == CONJUNCTIVITIS_INDEX:

            false_positives.append(
                {
                    "path": image_path,
                    "probability":
                        conjunctivitis_probability,
                }
            )


        if counter % 50 == 0:

            print(
                f"Checked "
                f"{counter}/"
                f"{len(healthy_images)}"
            )


# ============================================================
# SORT FALSE POSITIVES
# ============================================================

false_positives.sort(
    key=lambda item:
        item["probability"],
    reverse=True
)


print()
print(
    f"False positives found: "
    f"{len(false_positives)}"
)


if len(false_positives) == 0:

    print()
    print(
        "No false positives found."
    )

    raise SystemExit(0)


# Only use the first NUMBER_TO_SHOW.
selected_images = false_positives[
    :NUMBER_TO_SHOW
]


print()
print(
    f"Generating Grad-CAM for "
    f"{len(selected_images)} images."
)


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


# ============================================================
# HOOK FUNCTIONS
# ============================================================

def save_activation(
    module,
    input,
    output
):

    global activations

    activations = output


def save_gradient(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


# ============================================================
# TARGET LAYER
# ============================================================

# Final convolutional feature layer of EfficientNet-B0.
target_layer = model.features[-1]


forward_handle = (
    target_layer.register_forward_hook(
        save_activation
    )
)


backward_handle = (
    target_layer.register_full_backward_hook(
        save_gradient
    )
)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def create_gradcam(
    image_path,
    target_class
):

    global activations
    global gradients


    activations = None
    gradients = None


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    tensor = transform(
        original
    )


    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.zero_grad()


    output = model(
        tensor
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )


    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()


    predicted_probability = (
        probabilities[
            0,
            predicted_index
        ].item()
    )


    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    score = output[
        0,
        target_class
    ]


    score.backward()


    # --------------------------------------------------------
    # Make sure hooks worked
    # --------------------------------------------------------

    if activations is None:

        raise RuntimeError(
            "Could not capture activations."
        )


    if gradients is None:

        raise RuntimeError(
            "Could not capture gradients."
        )


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    feature_maps = activations[
        0
    ]


    gradient_maps = gradients[
        0
    ]


    # --------------------------------------------------------
    # Calculate channel weights
    # --------------------------------------------------------

    weights = gradient_maps.mean(
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = torch.zeros(
        feature_maps.shape[1:],
        device=DEVICE
    )


    for channel in range(
        feature_maps.shape[0]
    ):

        cam += (
            weights[channel]
            * feature_maps[channel]
        )


    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = torch.relu(
        cam
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    cam = (
        cam
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    minimum = cam.min()
    maximum = cam.max()


    if (
        maximum - minimum
        > 1e-8
    ):

        cam = (
            cam - minimum
        ) / (
            maximum - minimum
        )

    else:

        cam = np.zeros_like(
            cam
        )


    # --------------------------------------------------------
    # Convert to 0-255
    # --------------------------------------------------------

    cam_uint8 = (
        cam * 255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap = Image.fromarray(
        cam_uint8
    )


    heatmap = heatmap.resize(
        original.size,
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # Create simple heatmap
    # --------------------------------------------------------

    heat = np.array(
        heatmap
    )


    heatmap_rgb = np.zeros(
        (
            heat.shape[0],
            heat.shape[1],
            3
        ),
        dtype=np.uint8
    )


    # Red = strong activation
    heatmap_rgb[:, :, 0] = heat


    # Green = medium activation
    heatmap_rgb[:, :, 1] = (
        heat * 0.5
    ).astype(
        np.uint8
    )


    # Blue = low activation
    heatmap_rgb[:, :, 2] = (
        255 - heat
    )


    heatmap_image = Image.fromarray(
        heatmap_rgb
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    original_rgba = original.convert(
        "RGBA"
    )


    heatmap_rgba = heatmap_image.convert(
        "RGBA"
    )


    heatmap_rgba.putalpha(
        110
    )


    overlay = Image.alpha_composite(
        original_rgba,
        heatmap_rgba
    )


    return (
        original,
        heatmap_image,
        overlay,
        predicted_index,
        predicted_probability,
    )


# ============================================================
# CREATE REPORT IMAGES
# ============================================================

for number, item in enumerate(
    selected_images,
    start=1
):

    image_path = item["path"]


    print()
    print(
        f"Creating Grad-CAM "
        f"{number}/"
        f"{len(selected_images)}"
    )

    print(
        image_path.name
    )


    (
        original,
        heatmap,
        overlay,
        predicted_index,
        predicted_probability,
    ) = create_gradcam(
        image_path,
        CONJUNCTIVITIS_INDEX
    )


    # --------------------------------------------------------
    # Resize panels
    # --------------------------------------------------------

    panel_width = 450


    original_ratio = (
        original.height
        / original.width
    )


    panel_height = int(
        panel_width
        * original_ratio
    )


    original_panel = original.resize(
        (
            panel_width,
            panel_height
        )
    )


    heatmap_panel = heatmap.resize(
        (
            panel_width,
            panel_height
        )
    )


    overlay_panel = overlay.resize(
        (
            panel_width,
            panel_height
        )
    )


    # --------------------------------------------------------
    # Create canvas
    # --------------------------------------------------------

    canvas_width = (
        panel_width * 3
    )


    canvas_height = (
        panel_height + 80
    )


    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height
        ),
        "white"
    )


    # --------------------------------------------------------
    # Put images
    # --------------------------------------------------------

    canvas.paste(
        original_panel,
        (
            0,
            80
        )
    )


    canvas.paste(
        heatmap_panel,
        (
            panel_width,
            80
        )
    )


    canvas.paste(
        overlay_panel,
        (
            panel_width * 2,
            80
        )
    )


    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    draw = ImageDraw.Draw(
        canvas
    )


    font = ImageFont.load_default()


    draw.text(
        (
            10,
            20
        ),
        "Original",
        fill="black",
        font=font
    )


    draw.text(
        (
            panel_width + 10,
            20
        ),
        "Grad-CAM",
        fill="black",
        font=font
    )


    draw.text(
        (
            panel_width * 2 + 10,
            20
        ),
        "Overlay",
        fill="black",
        font=font
    )


    # --------------------------------------------------------
    # Add prediction information
    # --------------------------------------------------------

    prediction_text = (
        "True: NORMAL | "
        "Predicted: CONJUNCTIVITIS | "
        f"Probability: "
        f"{item['probability'] * 100:.1f}%"
    )


    draw.text(
        (
            10,
            canvas_height - 30
        ),
        prediction_text,
        fill="black",
        font=font
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_name = (
        f"{number:02d}_"
        f"{image_path.stem}_"
        f"conjunctivitis_"
        f"{item['probability'] * 100:.1f}pct.jpg"
    )


    output_path = (
        FALSE_POSITIVE_DIR
        / output_name
    )


    canvas.save(
        output_path,
        quality=95
    )


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()

backward_handle.remove()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("GRAD-CAM COMPLETE")
print("=" * 70)

print()
print(
    "Generated Grad-CAM images:"
)

print(
    FALSE_POSITIVE_DIR
)

print()
print(
    "Open that folder and look at the images."
)

print()
print(
    "Each image contains:"
)

print(
    "1. Original eye image"
)

print(
    "2. Grad-CAM heatmap"
)

print(
    "3. Heatmap over the original image"
)

print()
print("=" * 70)