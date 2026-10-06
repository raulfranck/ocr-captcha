import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ocr import load_image


class FakeOCR:
    device = "fake"

    def predict(self, images):
        return [f"{img.width}x{img.height}" for img in images]


def _png(size=(120, 40), mode="RGBA") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    app.state.ocr = FakeOCR()
    with TestClient(app) as c:
        yield c
    del app.state.ocr


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "device": "fake"}


def test_upload(client):
    r = client.post("/ocr", files={"file": ("c.png", _png(), "image/png")})
    assert r.status_code == 200
    assert r.json()["text"] == "120x40"


def test_base64_and_data_url(client):
    b64 = base64.b64encode(_png((50, 20), "RGB")).decode()
    assert client.post("/ocr/base64", json={"image": b64}).json()["text"] == "50x20"
    r = client.post("/ocr/base64", json={"image": f"data:image/png;base64,{b64}"})
    assert r.json()["text"] == "50x20"


def test_batch(client):
    files = [("files", (f"{i}.png", _png((10 + i, 10)), "image/png")) for i in range(3)]
    assert client.post("/ocr/batch", files=files).json()["texts"] == ["10x10", "11x10", "12x10"]


def test_rejects_non_image(client):
    r = client.post("/ocr", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_bad_base64(client):
    assert client.post("/ocr/base64", json={"image": "%%%"}).status_code == 400


def test_rejects_large_image(client, monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_BYTES", "10")
    app.state.ocr = FakeOCR()
    with TestClient(app) as c:
        r = c.post("/ocr", files={"file": ("c.png", _png(), "image/png")})
    assert r.status_code == 413


def test_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "segredo")
    app.state.ocr = FakeOCR()
    with TestClient(app) as c:
        files = {"file": ("c.png", _png(), "image/png")}
        assert c.post("/ocr", files=files).status_code == 401
        assert c.post("/ocr", files=files, headers={"X-API-Key": "segredo"}).status_code == 200
    del app.state.ocr


def test_16bit_grayscale_png_is_scaled_not_clipped():
    arr = np.full((20, 60), 65535, dtype=np.uint16)
    arr[5:15, 10:50] = 1000  # dark text on light background
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    assert Image.open(io.BytesIO(buf.getvalue())).mode.startswith("I;16")
    out = np.asarray(load_image(buf.getvalue()))
    assert out[10, 30].max() < 10 and out[0, 0].min() == 255
