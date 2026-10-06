from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


client = TestClient(app)


def create_test_image():
    """
    Create a small valid RGB JPEG image for API testing.
    """

    image = Image.new(
        "RGB",
        (224, 224),
        (120, 150, 180)
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG"
    )

    buffer.seek(0)

    return buffer


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["app"] == "NETRA AI"

    assert data["model"] == (
        "eyecheck_multiclass_4class_v2"
    )

    assert data["classes"] == [
        "Cataract",
        "conjunctivitis",
        "normal",
        "stye"
    ]


def test_predict_endpoint():

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "probability" in data
    assert "quality_score" in data
    assert "image_quality" in data
    assert "gradcam" in data
    assert "all_probabilities" in data

    assert data["label"] in [
        "Cataract",
        "conjunctivitis",
        "normal",
        "stye",
        "invalid_image"
    ]

    assert 0.0 <= data["probability"] <= 1.0


def test_predict_requires_file():

    response = client.post("/predict")

    assert response.status_code == 422