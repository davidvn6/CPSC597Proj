"""Meaningful upload/API checks with fake inference, not model validation.

Run: python pneumonia_detector/check_app.py
The fake engine is only used here, never by app.py's normal launch.
"""
import base64
import io
import json
import unittest
import zipfile

from PIL import Image
from app import create_app, png_bytes


class FakeEngine:
    """Deliberately synthetic output to exercise HTTP and download behavior."""
    epoch = 8
    threshold = 0.201475
    checkpoint_hash = "FAKE_TEST_CHECKPOINT"
    protocol_hash = "FAKE_TEST_PROTOCOL"
    device_label = "Fake engine — test only"

    def analyze(self, image):
        gray = image.resize((224, 224))
        pixels = png_bytes(gray)
        return {
            "prediction": "NORMAL", "score": 0.1, "threshold": self.threshold,
            "seconds": 0.01, "zero_map": True, "positive_map_max": 0.0,
            "input_png": pixels, "heatmap_png": pixels, "overlay_png": pixels,
            "heatmap_bytes": base64.b64encode(bytes(224 * 224)).decode(),
        }


class RouteChecks(unittest.TestCase):
    def setUp(self):
        self.app = create_app(FakeEngine())
        self.app.testing = True
        self.client = self.app.test_client()

    def upload(self, content, name="test.png"):
        return self.client.post("/api/analyze", data={
            "image": (io.BytesIO(content), name)
        }, content_type="multipart/form-data")

    def test_success_and_archive(self):
        result = self.upload(png_bytes(Image.new("L", (100, 80), 128)))
        self.assertEqual(result.status_code, 200)
        payload = result.get_json()
        self.assertEqual(payload["result"]["prediction"], "NORMAL")
        self.assertEqual(payload["result"]["original_width"], 100)
        self.assertEqual(len(base64.b64decode(payload["heatmap_values"])), 224 * 224)
        archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload["archive"])))
        self.assertEqual(set(archive.namelist()), {
            "model_input.png", "pneumonia_gradcam.png", "overlay.png",
            "result.json", "README.txt"
        })
        self.assertEqual(json.loads(archive.read("result.json"))["pneumonia_score"], 0.1)
        self.assertEqual(result.headers["Cache-Control"], "no-store")

    def test_invalid_contents(self):
        self.assertEqual(self.upload(b"not an image").status_code, 400)

    def test_invalid_extension(self):
        self.assertEqual(self.upload(b"not an image", "test.dcm").status_code, 400)

    def test_tiny_image(self):
        self.assertEqual(self.upload(png_bytes(Image.new("L", (8, 8)))).status_code, 400)

    def test_missing_file(self):
        self.assertEqual(self.client.post("/api/analyze").status_code, 400)

    def test_oversized_request(self):
        self.assertEqual(self.upload(b"x" * (13 * 1024 * 1024)).status_code, 413)

    def test_no_model(self):
        client = create_app().test_client()
        self.assertEqual(client.post("/api/analyze").status_code, 503)
        self.assertFalse(client.get("/api/model").get_json()["ready"])

    def test_page_and_metadata(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertTrue(self.client.get("/api/model").get_json()["ready"])
        with self.client.get("/static/app.js") as response:
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
