import os
import csv

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms, datasets


# ============================================================
# SETTINGS
# ============================================================

V2_MODEL = "models/conjunctivitis_efficientnet_b0_clean.pth"
V3_MODEL = "models/conjunctivitis_efficientnet_b0_v3.pth"

TRAIN_DIR = "data/clean_split/train"

NORMAL_DIR = "data/external/healthy_eye"
CONJUNCTIVITIS_DIR = "data/external/infected_eye"

OUTPUT_FILE = "reports/v2_vs_v3_comparison.csv"

EXCLUDED_FILE = "170.jpeg"

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("V2 vs V3 EXTERNAL PREDICTION COMPARISON")
print("=" * 70)

print()
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GET CLASS MAPPING
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR
)

class_mapping = train_dataset.class_to_idx

normal_index = class_mapping["normal"]

conjunctivitis_index = class_mapping["conjunctivitis"]

print("-" * 70)
print("CLASS MAPPING")
print("-" * 70)

print()

for name, index in class_mapping.items():
    print(f"{index} = {name}")

print()


# ============================================================
# LOAD MODEL FUNCTION
# ============================================================

def load_model(model_path):

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print("-" * 70)
print("LOADING MODELS")
print("-" * 70)

print()

print("Loading V2...")

v2_model = load_model(
    V2_MODEL
)

print("V2 loaded successfully.")

print()

print("Loading V3...")

v3_model = load_model(
    V3_MODEL
)

print("V3 loaded successfully.")

print()


# ============================================================
# FIND IMAGES
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def get_images(folder):

    image_paths = []

    for filename in os.listdir(folder):

        full_path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(full_path):
            continue

        if not filename.lower().endswith(
            valid_extensions
        ):
            continue

        if filename.lower() == EXCLUDED_FILE.lower():

            # This is the known duplicate.
            continue

        image_paths.append(
            full_path
        )

    return sorted(image_paths)


normal_images = get_images(
    NORMAL_DIR
)

conjunctivitis_images = get_images(
    CONJUNCTIVITIS_DIR
)


# ============================================================
# CREATE RECORDS
# ============================================================

records = []

for path in normal_images:

    records.append(
        {
            "path": path,
            "true_label": "normal",
            "true_index": normal_index
        }
    )


for path in conjunctivitis_images:

    records.append(
        {
            "path": path,
            "true_label": "conjunctivitis",
            "true_index": conjunctivitis_index
        }
    )


print("-" * 70)
print("EXTERNAL DATASET")
print("-" * 70)

print()

print(
    "Normal images:",
    len(normal_images)
)

print(
    "Conjunctivitis images:",
    len(conjunctivitis_images)
)

print(
    "Total images:",
    len(records)
)

print()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(model, image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

        probability = torch.softmax(
            output,
            dim=1
        )

    predicted_index = torch.argmax(
        probability,
        dim=1
    ).item()

    conjunctivitis_probability = (
        probability[
            0,
            conjunctivitis_index
        ].item()
    )

    return (
        predicted_index,
        conjunctivitis_probability
    )


# ============================================================
# RUN COMPARISON
# ============================================================

print("-" * 70)
print("COMPARING V2 AND V3")
print("-" * 70)

print()

results = []

v2_same_as_v3 = 0

v2_correct_v3_wrong = 0

v2_wrong_v3_correct = 0

both_correct = 0

both_wrong = 0


for counter, record in enumerate(
    records,
    start=1
):

    image_path = record["path"]

    true_index = record["true_index"]

    true_label = record["true_label"]

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        # -----------------------------
        # V2
        # -----------------------------

        v2_prediction, v2_probability = predict(
            v2_model,
            image
        )

        # -----------------------------
        # V3
        # -----------------------------

        v3_prediction, v3_probability = predict(
            v3_model,
            image
        )

        # -----------------------------
        # Names
        # -----------------------------

        v2_label = (
            "normal"
            if v2_prediction == normal_index
            else "conjunctivitis"
        )

        v3_label = (
            "normal"
            if v3_prediction == normal_index
            else "conjunctivitis"
        )

        # -----------------------------
        # Correct / wrong
        # -----------------------------

        v2_correct = (
            v2_prediction == true_index
        )

        v3_correct = (
            v3_prediction == true_index
        )

        if v2_prediction == v3_prediction:

            v2_same_as_v3 += 1

        if v2_correct and v3_correct:

            both_correct += 1

        elif not v2_correct and not v3_correct:

            both_wrong += 1

        elif v2_correct and not v3_correct:

            v2_correct_v3_wrong += 1

        elif not v2_correct and v3_correct:

            v2_wrong_v3_correct += 1

        results.append(
            {
                "filename":
                    os.path.basename(
                        image_path
                    ),

                "true_label":
                    true_label,

                "v2_prediction":
                    v2_label,

                "v2_conjunctivitis_probability":
                    round(
                        v2_probability,
                        6
                    ),

                "v2_correct":
                    v2_correct,

                "v3_prediction":
                    v3_label,

                "v3_conjunctivitis_probability":
                    round(
                        v3_probability,
                        6
                    ),

                "v3_correct":
                    v3_correct,

                "same_prediction":
                    (
                        v2_prediction
                        == v3_prediction
                    )
            }
        )

    except Exception as error:

        print()
        print(
            "ERROR:",
            image_path
        )

        print(
            error
        )

    if (
        counter % 50 == 0
        or counter == len(records)
    ):

        print(
            f"Checked "
            f"{counter}/"
            f"{len(records)}"
        )


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    "reports",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "true_label",
            "v2_prediction",
            "v2_conjunctivitis_probability",
            "v2_correct",
            "v3_prediction",
            "v3_conjunctivitis_probability",
            "v3_correct",
            "same_prediction"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

total = len(results)

agreement_percentage = (
    100.0
    * v2_same_as_v3
    / total
)

print()
print("=" * 70)
print("V2 vs V3 COMPARISON COMPLETE")
print("=" * 70)

print()

print(
    f"Total images compared: {total}"
)

print(
    f"V2 and V3 made the same prediction: "
    f"{v2_same_as_v3}/{total} "
    f"({agreement_percentage:.2f}%)"
)

print()

print(
    "Both models correct:",
    both_correct
)

print(
    "Both models wrong:",
    both_wrong
)

print()

print(
    "V2 correct, V3 wrong:",
    v2_correct_v3_wrong
)

print(
    "V2 wrong, V3 correct:",
    v2_wrong_v3_correct
)

print()

print(
    "Comparison saved to:"
)

print(
    OUTPUT_FILE
)

print()

print("=" * 70)