r"""PneumoniaDetector: local CPSC597 Flask demonstration.

Extract pneumonia_detector/ into CPSC597Proj/, then run:
    .\.venv\Scripts\python.exe .\pneumonia_detector\app.py

The recorded checkpoint and threshold are verified at startup. Uploads are
processed in memory; this app does not train or change the research models.
"""

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
import time
import warnings
import zipfile

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CPSC597_PROJECT_ROOT", APP_DIR.parent)).resolve()
PROTOCOL_REL = Path("results/threshold_selection/20261008_182555_424842/selected_thresholds.json")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def sha256(path):
    """Hash a checkpoint in chunks rather than loading another full copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def decode_upload(upload):
    """Validate actual image contents, not just the supplied extension."""
    if upload is None or not upload.filename:
        raise ValueError("Choose a chest X-ray image first.")
    if Path(upload.filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Please use a JPG, JPEG, or PNG image. DICOM is not supported.")
    content = upload.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a nonempty image smaller than 12 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("The file contents must be JPEG or PNG.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Please choose a single-frame image.")
                source.load()
                # Match XRayDataset: apply EXIF orientation and convert to L.
                image = ImageOps.exif_transpose(source).convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError):
        raise ValueError("This image could not be read, or its dimensions are too large.")
    if min(image.size) < 32:
        raise ValueError("The image must be at least 32 pixels on each side.")
    return image, secure_filename(upload.filename) or "xray.png"


def colorize(values, palette="ember"):
    """Use the same small color lookup table as the browser renderer."""
    import numpy as np
    anchors = {
        "ember": [[18, 23, 57], [50, 79, 156], [24, 160, 182],
                  [253, 194, 87], [244, 79, 57]],
        "viridis": [[68, 1, 84], [59, 82, 139], [33, 145, 140],
                    [94, 201, 98], [253, 231, 37]],
    }[palette]
    stops = np.linspace(0, 255, len(anchors))
    return np.stack([
        np.interp(values, stops, [a[channel] for a in anchors])
        for channel in range(3)
    ], axis=-1).round().astype("uint8")


class InferenceEngine:
    """Load one frozen model; serialize requests to protect Grad-CAM hooks."""

    def __init__(self, project_root):
        import torch
        from torch import nn
        from torchvision.models import resnet50
        from torchvision.transforms import v2

        self.torch = torch
        self.lock = threading.Lock()
        protocol_path = project_root / PROTOCOL_REL
        if not protocol_path.exists():
            raise RuntimeError(f"Missing frozen threshold protocol: {protocol_path}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        entry = protocol["models"]["ResNet50_square"]
        if protocol.get("test_set_used") is not False:
            raise RuntimeError("The threshold protocol must record validation-only selection.")
        checkpoint_path = (project_root / entry["checkpoint"]).resolve()
        if not checkpoint_path.is_relative_to(project_root):
            raise RuntimeError("Checkpoint must be inside the project folder.")
        if sha256(checkpoint_path) != entry["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash differs from the frozen threshold protocol.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config.get("architecture") != "ResNet50CNN":
            raise RuntimeError("Expected the recorded ResNet50CNN checkpoint.")
        if int(checkpoint["epoch"]) != int(entry["checkpoint_epoch"]):
            raise RuntimeError("Checkpoint epoch differs from the threshold protocol.")
        if int(config["image_size"]) != 224:
            raise RuntimeError("This app expects the recorded 224-pixel model input.")
        if entry.get("preprocessing") != "square_resize_imagenet_normalization":
            raise RuntimeError("Expected square-resize preprocessing.")

        # Recreate ResNet50CNN exactly, without importing training scripts
        # or downloading pretrained weights. The checkpoint supplies weights.
        class FrozenResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = resnet50(weights=None)
                self.network.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 1))

            def forward(self, images):
                return self.network(images).squeeze(1)

        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FrozenResNet()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        # Input gradients still allow Grad-CAM while weight gradients are disabled.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.input_transform = v2.Compose([
            v2.Grayscale(num_output_channels=3), v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.normalize = v2.Normalize(
            mean=config["normalization_mean"], std=config["normalization_std"]
        )
        self.threshold = float(entry["threshold"])
        if not 0 <= self.threshold <= 1:
            raise RuntimeError("Invalid recorded decision threshold.")
        self.batch_size = int(config["batch_size"])
        self.epoch = int(checkpoint["epoch"])
        self.checkpoint_hash = entry["checkpoint_sha256"]
        self.protocol_hash = sha256(protocol_path)
        bundled = json.loads((APP_DIR / "model_metrics.json").read_text(encoding="utf-8"))
        if abs(self.threshold - float(bundled["threshold"])) > 1e-12:
            raise RuntimeError("Recorded threshold differs from the bundled metrics snapshot.")
        self.device_label = (torch.cuda.get_device_name(0)
                             if self.device.type == "cuda" else "CPU")

    def analyze(self, image):
        """Predict and explain the pneumonia logit in the same forward pass."""
        import numpy as np
        torch = self.torch
        started = time.perf_counter()
        with self.lock, torch.enable_grad():
            pixels = self.input_transform(image)
            # Repeat the input to retain the evaluation batch shape. This reduces
            # batch-size-related numeric drift seen in the earlier Grad-CAM run.
            # Only image 0's logit is differentiated and displayed.
            batch = self.normalize(pixels.clone()).unsqueeze(0).repeat(
                self.batch_size, 1, 1, 1
            ).to(self.device).detach().requires_grad_(True)
            captured = {}

            def capture(module, inputs, output):
                captured["features"] = output

            hook = self.model.network.layer4.register_forward_hook(capture)
            try:
                logits = self.model(batch).reshape(-1)
                features = captured["features"]
                gradients = torch.autograd.grad(logits[0], features)[0]
                weights = gradients[0:1].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * features[0:1]).sum(dim=1, keepdim=True))
                positive_max = float(cam.max().detach().cpu())
                cam = torch.nn.functional.interpolate(
                    cam, size=(224, 224), mode="bilinear", align_corners=False
                )[0, 0].detach().cpu().numpy()
                if float(cam.max()) > 0:
                    cam /= float(cam.max())
                else:
                    cam = np.zeros_like(cam)
                score = float(torch.sigmoid(logits[0]).detach().cpu())
            finally:
                hook.remove()
        if not math.isfinite(score):
            raise RuntimeError("The model returned a non-finite score.")
        gray = (pixels[0].cpu().numpy() * 255).round().astype("uint8")
        quantized = (cam * 255).round().astype("uint8")
        heat = colorize(quantized)
        opacity = 0.55 * (quantized.astype(float) / 255)[..., None]
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        overlay = (rgb * (1 - opacity) + heat * opacity).round().astype("uint8")
        return {
            "score": score, "prediction": "PNEUMONIA" if score >= self.threshold else "NORMAL",
            "threshold": self.threshold, "seconds": time.perf_counter() - started,
            "zero_map": positive_max == 0, "positive_map_max": positive_max,
            "input_png": png_bytes(Image.fromarray(gray)),
            "heatmap_png": png_bytes(Image.fromarray(heat)),
            "overlay_png": png_bytes(Image.fromarray(overlay)),
            "heatmap_bytes": base64.b64encode(quantized.tobytes()).decode("ascii"),
        }


def create_app(engine=None, project_root=PROJECT_ROOT):
    """An injected engine permits route checks without the private checkpoint."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["ENGINE"] = engine
    app.config["PROJECT_ROOT"] = project_root
    # These are the actual held-out metrics supplied in this project.
    metrics = json.loads((APP_DIR / "model_metrics.json").read_text(encoding="utf-8"))

    @app.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "script-src 'self'; style-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/model")
    def model_info():
        current = app.config["ENGINE"]
        return jsonify({
            "ready": current is not None,
            "model": "ResNet-50", "variant": "Square resize",
            "epoch": current.epoch if current else 8,
            "threshold": current.threshold if current else None,
            "device": current.device_label if current else "Preview only",
            "metrics": metrics,
        })

    @app.post("/api/analyze")
    def analyze():
        current = app.config["ENGINE"]
        if current is None:
            return jsonify(error="Model not loaded. Start the app from your project folder."), 503
        try:
            image, filename = decode_upload(request.files.get("image"))
            result = current.analyze(image)
            metadata = {
                "filename": filename, "original_width": image.width,
                "original_height": image.height, "model": "ResNet50_square",
                "checkpoint_epoch": current.epoch,
                "checkpoint_sha256": current.checkpoint_hash,
                "threshold_protocol_sha256": current.protocol_hash,
                "prediction": result["prediction"], "pneumonia_score": result["score"],
                "threshold": result["threshold"], "inference_seconds": result["seconds"],
                "input_size": [224, 224], "preprocessing": "grayscale, square resize, ImageNet normalization",
                "gradcam_target": "PNEUMONIA logit", "gradcam_layer": "network.layer4",
                "zero_positive_map": result["zero_map"],
                "held_out_model_metrics": metrics,
                "note": "Research demonstration. Score is not a calibrated clinical probability. Grad-CAM is not verified pneumonia localization.",
            }
            # The archive exists only in memory and is returned to this browser.
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
                package.writestr("model_input.png", result["input_png"])
                package.writestr("pneumonia_gradcam.png", result["heatmap_png"])
                package.writestr("overlay.png", result["overlay_png"])
                package.writestr("result.json", json.dumps(metadata, indent=2))
                package.writestr("README.txt", "Heatmaps explain the pneumonia logit, including for NORMAL predictions. Each map is normalized independently. Overlay uses the square model input. Default overlay opacity: 55%, weighted by heatmap value.\n")
            return jsonify({
                "result": metadata, "original": data_url(png_bytes(image)),
                "model_input": data_url(result["input_png"]),
                "heatmap": data_url(result["heatmap_png"]),
                "overlay": data_url(result["overlay_png"]),
                "heatmap_values": result["heatmap_bytes"],
                "archive": base64.b64encode(archive.getvalue()).decode("ascii"),
            })
        except RequestEntityTooLarge:
            raise
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception("Image analysis failed")
            return jsonify(error="Analysis failed. Check the terminal for details and try another image."), 500

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="The upload is too large. Choose an image smaller than 12 MB."), 413

    return app


if __name__ == "__main__":
    print("Loading the frozen square-trained ResNet-50...")
    try:
        engine = InferenceEngine(PROJECT_ROOT)
    except Exception as error:
        raise SystemExit(f"Could not start PneumoniaDetector: {error}\nSee pneumonia_detector/README.md for setup.")
    print(f"Model: ResNet-50, epoch {engine.epoch} | threshold {engine.threshold:.8f}")
    print(f"Device: {engine.device_label}")
    print("Open http://127.0.0.1:5000 in your browser. Ctrl+C stops the app.")
    # Local demonstration only. Disable reloader to avoid loading weights twice.
    create_app(engine).run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=False)
