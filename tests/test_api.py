import io
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)


def _create_test_image_bytes(width=256, height=256, color=(34, 139, 34)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_valid_image_prediction():
    img_bytes = _create_test_image_bytes()
    res = client.post(
        "/predict",
        files={"file": ("leaf_sample.jpg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data["prediction"]
    assert data["prediction"]["input_shape"] == [1, 224, 224, 3]


def test_what_if_invalid_mime_type():
    res = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert res.status_code == 415


def test_what_if_corrupted_image_bytes():
    res = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"not-a-real-jpeg-stream", "image/jpeg")},
    )
    assert res.status_code == 400


def test_what_if_image_too_small():
    tiny_img = _create_test_image_bytes(width=16, height=16)
    res = client.post(
        "/predict",
        files={"file": ("tiny.jpg", tiny_img, "image/jpeg")},
    )
    assert res.status_code == 422
