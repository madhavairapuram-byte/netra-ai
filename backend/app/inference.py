# =========================================================
# NETRA AI
# Inference + CLIP Eye Validation + 4-Class EfficientNet
# =========================================================

from dataclasses import dataclass
from pathlib import Path
import base64

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import models, transforms
from transformers import pipeline


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "eyecheck_multiclass_4class_v2.pth"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# PREDICTION RESULT
# =========================================================

@dataclass
class Prediction:

    probability: float
    label: str

    image_quality: str
    quality_score: float

    method: str
    note: str

    explanation_image: str | None = None

    all_probabilities: dict[str, float] | None = None


# =========================================================
# 4-CLASS IMAGE PREPROCESSING
# =========================================================

MODEL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((56, 56)),
        transforms.Resize((224, 224)),
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


# =========================================================
# CLIP EYE VALIDATION
# =========================================================

CLIP_MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)

CLIP_LABELS = [
    "an image of a human eye",
    "an image that does not contain a human eye",
]

CLIP_EYE_THRESHOLD = 0.50

_CLIP_CLASSIFIER = None


def _load_clip():

    global _CLIP_CLASSIFIER

    if _CLIP_CLASSIFIER is not None:
        return _CLIP_CLASSIFIER

    # -----------------------------------------------------
    # IMPORTANT:
    # Keep CLIP on CPU.
    #
    # EfficientNet uses the RTX 3050.
    # This avoids unnecessary GPU memory usage.
    # -----------------------------------------------------

    _CLIP_CLASSIFIER = pipeline(
        "zero-shot-image-classification",
        model=CLIP_MODEL_NAME,
        device=-1,
    )

    return _CLIP_CLASSIFIER


def _check_is_eye(
    image: Image.Image,
):

    classifier = _load_clip()

    results = classifier(
        image,
        candidate_labels=CLIP_LABELS,
    )

    eye_score = 0.0

    for result in results:

        if (
            result["label"]
            == "an image of a human eye"
        ):

            eye_score = float(
                result["score"]
            )

            break

    is_eye = (
        eye_score
        >= CLIP_EYE_THRESHOLD
    )

    return (
        is_eye,
        eye_score,
    )


# =========================================================
# 4-CLASS MODEL
# =========================================================

_MODEL = None


CLASS_NAMES = [
    "Cataract",
    "conjunctivitis",
    "normal",
    "stye",
]


def _load_model(
    model_path: str | None = None,
):

    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if model_path is None:
        model_path = str(
            DEFAULT_MODEL_PATH
        )

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    # -----------------------------------------------------
    # Create EfficientNet-B0
    # -----------------------------------------------------

    model = models.efficientnet_b0(
        weights=None
    )

    # -----------------------------------------------------
    # Four output classes
    # -----------------------------------------------------

    model.classifier[1] = (
        torch.nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES),
        )
    )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    # -----------------------------------------------------
    # Extract state dictionary
    # -----------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # -----------------------------------------------------
    # Remove DataParallel prefix
    # -----------------------------------------------------

    cleaned_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    # -----------------------------------------------------
    # Load weights
    # -----------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    _MODEL = model

    return _MODEL


# =========================================================
# IMAGE QUALITY
# =========================================================

def _quality_score(
    image: Image.Image,
):

    image_rgb = np.array(
        image
    )

    image_gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    # -----------------------------------------------------
    # Sharpness
    # -----------------------------------------------------

    sharpness = float(
        cv2.Laplacian(
            image_gray,
            cv2.CV_64F,
        ).var()
    )

    sharpness_score = min(
        sharpness / 150.0,
        1.0,
    )

    # -----------------------------------------------------
    # Brightness
    # -----------------------------------------------------

    brightness = float(
        image_gray.mean()
    )

    if brightness < 35:

        brightness_score = (
            brightness / 35.0
        )

    elif brightness > 220:

        brightness_score = (
            (255.0 - brightness)
            / 35.0
        )

    else:

        brightness_score = 1.0

    brightness_score = max(
        0.0,
        min(
            brightness_score,
            1.0,
        ),
    )

    # -----------------------------------------------------
    # Combined quality
    # -----------------------------------------------------

    quality = (
        0.6 * sharpness_score
        + 0.4 * brightness_score
    )

    return float(
        quality
    )


# =========================================================
# 4-CLASS MODEL PREDICTION
# =========================================================

def _model_prediction(
    image: Image.Image,
    model,
):

    tensor = (
        MODEL_TRANSFORM(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                output,
                dim=1,
            )
        )

    predicted_index = int(
        torch.argmax(
            probabilities,
            dim=1,
        ).item()
    )

    predicted_probability = float(
        probabilities[
            0,
            predicted_index,
        ].item()
    )

    predicted_label = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    all_probabilities = {
        CLASS_NAMES[index]:
            float(
                probabilities[
                    0,
                    index,
                ].item()
            )
        for index in range(
            len(CLASS_NAMES)
        )
    }

    return (
        predicted_label,
        predicted_probability,
        all_probabilities,
        predicted_index,
    )


# =========================================================
# GRAD-CAM
# =========================================================

def _generate_gradcam(
    image: Image.Image,
    model,
    target_class: int,
):

    activations = []
    gradients = []

    target_layer = (
        model.features[-1]
    )

    def forward_hook(
        module,
        input,
        output,
    ):

        activations.append(
            output.detach()
        )

    def backward_hook(
        module,
        grad_input,
        grad_output,
    ):

        if (
            grad_output
            and grad_output[0]
            is not None
        ):

            gradients.append(
                grad_output[0].detach()
            )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        tensor = (
            MODEL_TRANSFORM(
                image
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        model.zero_grad(
            set_to_none=True
        )

        output = model(
            tensor
        )

        target = output[
            0,
            target_class,
        ]

        target.backward()

        if (
            not activations
            or not gradients
        ):

            return None

        activation = (
            activations[0][0]
        )

        gradient = (
            gradients[0][0]
        )

        weights = gradient.mean(
            dim=(1, 2),
            keepdim=True,
        )

        cam = (
            weights
            * activation
        ).sum(
            dim=0
        )

        cam = torch.relu(
            cam
        )

        cam = cam.detach().cpu().numpy()

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        if (
            cam_max
            > cam_min
        ):

            cam = (
                cam - cam_min
            ) / (
                cam_max
                - cam_min
            )

        else:

            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        original = np.array(
            image.convert("RGB")
        )

        height, width = (
            original.shape[:2]
        )

        cam = cv2.resize(
            cam,
            (
                width,
                height,
            ),
        )

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = (
            cv2.applyColorMap(
                heatmap,
                cv2.COLORMAP_JET,
            )
        )

        original_bgr = (
            cv2.cvtColor(
                original,
                cv2.COLOR_RGB2BGR,
            )
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.60,
            heatmap,
            0.40,
            0,
        )

        ok, encoded = (
            cv2.imencode(
                ".jpg",
                overlay,
            )
        )

        if not ok:

            return None

        encoded_string = (
            base64.b64encode(
                encoded.tobytes()
            ).decode(
                "utf-8"
            )
        )

        return (
            "data:image/jpeg;base64,"
            + encoded_string
        )

    finally:

        forward_handle.remove()

        backward_handle.remove()


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict(
    image: Image.Image,
    model_path: str | None = None,
):

    image = image.convert(
        "RGB"
    )

    # =====================================================
    # STEP 1 — CLIP EYE VALIDATION
    # =====================================================

    is_eye, eye_score = (
        _check_is_eye(
            image
        )
    )

    if not is_eye:

        return Prediction(

            probability=0.0,

            label="invalid_image",

            image_quality=(
                "not_assessed"
            ),

            quality_score=0.0,

            method=(
                "clip_eye_validation"
            ),

            note=(
                "The uploaded image "
                "does not appear to "
                "contain a human eye. "
                "Please upload a clear "
                "photograph of an eye."
            ),

            explanation_image=None,

            all_probabilities=None,
        )

    # =====================================================
    # STEP 2 — IMAGE QUALITY
    # =====================================================

    quality_score = (
        _quality_score(
            image
        )
    )

    if quality_score < 0.20:

        image_quality = "poor"

    elif quality_score < 0.45:

        image_quality = "fair"

    else:

        image_quality = "good"

    # =====================================================
    # STEP 3 — LOAD 4-CLASS MODEL
    # =====================================================

    model = _load_model(
        model_path
    )

    # =====================================================
    # STEP 4 — 4-CLASS PREDICTION
    # =====================================================

    (
        label,
        probability,
        all_probabilities,
        predicted_index,
    ) = _model_prediction(
        image,
        model,
    )

    # =====================================================
    # STEP 5 — GRAD-CAM
    #
    # Grad-CAM is optional.
    # A Grad-CAM failure must never
    # prevent the prediction itself.
    # =====================================================

    explanation_image = None

    try:

        explanation_image = (
            _generate_gradcam(
                image,
                model,
                predicted_index,
            )
        )

    except Exception:

        explanation_image = None

    # =====================================================
    # STEP 6 — RETURN RESULT
    # =====================================================

    return Prediction(

        probability=probability,

        label=label,

        image_quality=(
            image_quality
        ),

        quality_score=(
            quality_score
        ),

        method=(
            "clip_eye_validation_"
            "4class_efficientnet"
        ),

        note=(
            "The image passed "
            "the eye-image "
            "validation step and "
            "was analysed using "
            "the 4-class "
            "EfficientNet-B0 "
            "model."
        ),

        explanation_image=(
            explanation_image
        ),

        all_probabilities=(
            all_probabilities
        ),
    )