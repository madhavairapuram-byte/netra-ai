from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .inference import predict


app = FastAPI(
    title="NETRA AI",
    description="Educational eye-image screening prototype",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "NETRA AI",
        "model": "eyecheck_multiclass_4class_v2",
        "classes": [
            "Cataract",
            "conjunctivitis",
            "normal",
            "stye",
        ],
        "clip": True,
        "gradcam": True,
        "screening_mode": "single_image",
        "medical_validation": False,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the image: {exc}",
        )

    try:
        result = predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model file was not found: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        )

    return {
        "label": result.label,
        "probability": result.probability,
        "quality_score": result.quality_score,
        "image_quality": result.image_quality,
        "gradcam": result.explanation_image,
        "all_probabilities": result.all_probabilities,
        "method": result.method,
        "note": result.note,
        "quality_ok": result.image_quality != "poor",
        "clip_valid": result.label != "invalid_image",
        "clip_eye_probability": (
            None if result.label != "invalid_image" else 0.0
        ),
    }