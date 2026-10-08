"""
gradcam_resnet50.py

Generate pneumonia-target Grad-CAM images for the trained ResNet-50.

Selects:
    1. The same validation images used for InitialCNN Grad-CAM.
    2. Up to three additional ResNet-50 false negatives.
    3. Up to three additional ResNet-50 false positives.

Run from CPSC597Proj:
    .\\.venv\\Scripts\\python.exe .\\gradcam_resnet50.py

Outputs:
    results/resnet50/<training_run>/gradcam_resnet50/

No retraining is performed. The test set is not used.
"""

import hashlib
from pathlib import Path

import matplotlib

# Save figures without opening interactive plot windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import v2

from train_resnet50 import ResNet50CNN


# ------------------------------------------------------------
# Project settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "resnet50"

SEED = 42
EXTRA_EXAMPLES_PER_ERROR = 3

# Allow small score differences between batched evaluation and
# single-image Grad-CAM. Classification must still agree.
SCORE_TOLERANCE = 0.005


def sha256_file(path):
    """Calculate a SHA-256 hash using small file chunks."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_training_run():
    """Choose the most recent run containing a best-model checkpoint."""
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit(
            "No checkpoint found in results/resnet50.\n"
            "Run train_resnet50.py first."
        )

    return checkpoints[-1].parent


def get_validation_hash(config):
    """Find the saved validation hash using supported key names."""
    hashes = config.get("split_file_sha256")

    if not isinstance(hashes, dict):
        raise SystemExit(
            "The checkpoint does not contain a split_file_sha256 dictionary."
        )

    accepted_names = {
        "val",
        "val.csv",
        "validation",
        "validation.csv",
        "val_csv",
        "validation_csv",
    }

    matches = []

    for key, value in hashes.items():
        # Support simple names and paths ending in val.csv.
        normalized = str(key).replace("\\", "/").split("/")[-1].lower()

        if normalized in accepted_names:
            matches.append(str(value))

    if not matches:
        raise SystemExit(
            "Could not locate the validation hash in the checkpoint.\n"
            f"Available split hash keys: {list(hashes.keys())}"
        )

    if len(set(matches)) != 1:
        raise SystemExit(
            "The checkpoint contains conflicting validation hashes."
        )

    return matches[0]


def generate_gradcam(model, image_tensor):
    """
    Explain the pneumonia logit using the final residual block.

    Retain positive contributions and normalize each map for display.
    """
    captured = {}

    def capture_features(module, inputs, output):
        captured["features"] = output

    # Capture the output feature maps from the final residual block.
    hook = model.network.layer4.register_forward_hook(capture_features)

    try:
        # Grad-CAM requires gradients even when the model is in eval mode.
        with torch.enable_grad():
            image_tensor = image_tensor.detach().requires_grad_(True)
            logits = model(image_tensor).reshape(-1)

            features = captured["features"]

            # Measure how the pneumonia logit changes with each feature.
            gradients = torch.autograd.grad(
                outputs=logits[0],
                inputs=features,
            )[0]

            # Obtain one importance weight per feature channel.
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Combine feature maps and retain positive contributions.
            positive_map = torch.relu(
                (weights * features).sum(dim=1, keepdim=True)
            )

            raw_max = float(positive_map.max().detach().cpu())

            # Match the heatmap dimensions to the model input.
            resized_map = F.interpolate(
                positive_map,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            heatmap = resized_map[0, 0].detach().cpu().numpy()
            display_max = float(heatmap.max())

            if display_max > 0:
                heatmap = heatmap / display_max
            else:
                heatmap = np.zeros_like(heatmap)

            score = float(torch.sigmoid(logits[0]).detach().cpu())

        return heatmap, score, raw_max

    finally:
        # Remove the hook so it does not accumulate between images.
        hook.remove()


def save_figure(row, image, heatmap, score, output_path):
    """Save the resized input, heatmap, and overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Resized model input")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Pneumonia Grad-CAM")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.5 * heatmap,
    )
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{row['file']}\n"
        f"Actual: {row['label']} | "
        f"Saved prediction: {row['predicted_label']} | "
        f"Outcome: {row['outcome']}\n"
        f"Saved score: {float(row['pneumonia_score']):.4f} | "
        f"Grad-CAM score: {score:.4f}",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    # --------------------------------------------------------
    # Load the checkpoint and training configuration
    # --------------------------------------------------------

    run = select_training_run()
    print(f"Selected training run: {run}")

    checkpoint = torch.load(
        run / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    if config.get("architecture") != "ResNet50CNN":
        raise SystemExit(
            "The selected checkpoint is not a ResNet50CNN checkpoint."
        )

    # --------------------------------------------------------
    # Verify the validation assignments and predictions
    # --------------------------------------------------------

    validation_path = run / "val.csv"
    prediction_path = run / "validation_evaluation" / "predictions.csv"

    if not validation_path.exists():
        raise SystemExit(f"Validation CSV not found:\n{validation_path}")

    if not prediction_path.exists():
        raise SystemExit(
            "Validation predictions are missing.\n"
            "Run evalmetric_resnet50.py first."
        )

    expected_hash = get_validation_hash(config)

    if sha256_file(validation_path) != expected_hash:
        raise SystemExit(
            "The saved validation CSV does not match the training configuration."
        )

    validation = pd.read_csv(validation_path)
    predictions = pd.read_csv(prediction_path)

    required_columns = {
        "path",
        "file",
        "label",
        "target",
        "pneumonia_score",
        "predicted_target",
        "predicted_label",
        "outcome",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise SystemExit(
            f"Missing prediction columns: {sorted(missing)}"
        )

    if not validation["path"].is_unique:
        raise SystemExit("Duplicate paths found in the validation CSV.")

    if not predictions["path"].is_unique:
        raise SystemExit("Duplicate paths found in validation predictions.")

    if set(predictions["path"]) != set(validation["path"]):
        raise SystemExit(
            "Predictions do not match the saved validation image assignments."
        )

    # --------------------------------------------------------
    # Locate InitialCNN's selected Grad-CAM examples
    # --------------------------------------------------------

    initial_run_value = config.get("initial_run")

    if not initial_run_value:
        raise SystemExit(
            "The checkpoint configuration does not contain initial_run."
        )

    initial_run = Path(initial_run_value)

    # Support an absolute path or a training-run folder name.
    if not initial_run.is_absolute():
        initial_run = ROOT / "results" / "initial" / initial_run

    initial_selection_path = (
        initial_run / "gradcam_initial" / "selected_examples.csv"
    )

    if not initial_selection_path.exists():
        raise SystemExit(
            "InitialCNN example selection not found:\n"
            f"{initial_selection_path}"
        )

    initial_selection = pd.read_csv(initial_selection_path)

    if "path" not in initial_selection.columns:
        raise SystemExit(
            "The InitialCNN selected_examples.csv has no path column."
        )

    # --------------------------------------------------------
    # Select shared images and additional ResNet-50 errors
    # --------------------------------------------------------

    predictions_by_path = predictions.set_index("path", drop=False)

    selected_rows = []
    used_paths = set()

    for image_path in initial_selection["path"]:
        if image_path in used_paths:
            continue

        if image_path not in predictions_by_path.index:
            raise SystemExit(
                "An InitialCNN example is missing from validation:\n"
                f"{image_path}"
            )

        # Use ResNet-50's saved outcome for each shared image.
        row = predictions_by_path.loc[image_path].to_dict()
        row["selection_reason"] = "same_image_as_initial"

        selected_rows.append(row)
        used_paths.add(image_path)

    for outcome in ["FN", "FP"]:
        candidates = predictions[
            (predictions["outcome"] == outcome)
            & (~predictions["path"].isin(used_paths))
        ].sort_values("path")

        number = min(EXTRA_EXAMPLES_PER_ERROR, len(candidates))

        if number == 0:
            continue

        # A fixed seed makes the additional example selection reproducible.
        sampled = candidates.sample(
            n=number,
            random_state=SEED,
        ).sort_values("path")

        for _, row in sampled.iterrows():
            record = row.to_dict()
            record["selection_reason"] = f"additional_resnet50_{outcome}"

            selected_rows.append(record)
            used_paths.add(record["path"])

    selected = pd.DataFrame(selected_rows)

    if selected.empty:
        raise SystemExit("No images were selected.")

    # --------------------------------------------------------
    # Recreate validation preprocessing
    # --------------------------------------------------------

    image_size = int(config["image_size"])
    threshold = float(config["decision_threshold"])

    # Preserve an unnormalized copy for displaying the model input.
    input_transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.Resize(
            (image_size, image_size),
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])

    normalize = v2.Normalize(
        mean=config["normalization_mean"],
        std=config["normalization_std"],
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # The checkpoint supplies all model weights.
    model = ResNet50CNN(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output = run / "gradcam_resnet50"
    output.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Generating {len(selected)} examples using {device}.")

    # --------------------------------------------------------
    # Generate figures and verify score consistency
    # --------------------------------------------------------

    records = []

    for index, row in selected.iterrows():
        image_path = ROOT / row["path"]

        if not image_path.exists():
            raise SystemExit(f"Image not found:\n{image_path}")

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            unnormalized = input_transform(image)

        displayed_image = unnormalized[0].cpu().numpy()

        model_input = normalize(
            unnormalized.clone()
        ).unsqueeze(0).to(device)

        heatmap, score, raw_max = generate_gradcam(
            model,
            model_input,
        )

        saved_score = float(row["pneumonia_score"])
        difference = abs(score - saved_score)

        # Reject non-finite scores or differences above the tolerance.
        if not np.isfinite(score) or difference > SCORE_TOLERANCE:
            raise SystemExit(
                f"Score mismatch for {row['file']}:\n"
                f"Saved score: {saved_score:.8f}\n"
                f"Recomputed score: {score:.8f}\n"
                f"Absolute difference: {difference:.8f}\n"
                f"Allowed difference: {SCORE_TOLERANCE}\n"
                "Check the checkpoint and evaluation preprocessing."
            )

        if difference > 0.0001:
            print(
                f"Small score difference for {row['file']}: "
                f"{difference:.8f}"
            )

        recomputed_prediction = int(score >= threshold)

        # Even a small score difference must not change the classification.
        if recomputed_prediction != int(row["predicted_target"]):
            raise SystemExit(
                f"Prediction changed for {row['file']}:\n"
                f"Saved score: {saved_score:.8f}\n"
                f"Recomputed score: {score:.8f}\n"
                f"Threshold: {threshold}\n"
                "Check evaluation consistency before continuing."
            )

        stem = (
            f"{index + 1:02d}_{row['outcome']}_"
            f"{Path(row['file']).stem}"
        )

        figure_path = output / f"{stem}.png"

        save_figure(
            row,
            displayed_image,
            heatmap,
            score,
            figure_path,
        )

        # Save numerical maps alongside the visualizations.
        np.save(output / f"{stem}.npy", heatmap)

        record = row.to_dict()
        record.update({
            "gradcam_target": "PNEUMONIA",
            "recomputed_score": score,
            "score_difference": difference,
            "recomputed_prediction": recomputed_prediction,
            "raw_positive_map_max": raw_max,
            "zero_positive_map": raw_max == 0,
            "figure": figure_path.name,
        })

        records.append(record)

        print(f"Saved {row['outcome']}: {Path(row['file']).stem}")

    # --------------------------------------------------------
    # Save metadata and interpretation notes
    # --------------------------------------------------------

    pd.DataFrame(records).to_csv(
        output / "selected_examples.csv",
        index=False,
    )

    largest_difference = max(
        record["score_difference"] for record in records
    )

    summary = "\n".join([
        f"Selected training run: {run}",
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Examples generated: {len(records)}",
        "Target layer: network.layer4",
        "Target score: PNEUMONIA logit for every image",
        f"Decision threshold: {threshold}",
        f"Score tolerance: {SCORE_TOLERANCE}",
        f"Largest absolute score difference: {largest_difference:.8f}",
        "All recomputed classifications match the saved predictions.",
        "",
        "Shared examples match the InitialCNN image selection.",
        "Additional examples come from remaining ResNet-50 errors.",
        "Outcome labels come from the saved validation evaluation.",
        "",
        "Each heatmap is normalized independently.",
        "Heatmap intensity is not comparable across images or models.",
        "The models use different target layers and architectures.",
        "A zero map means no positive Grad-CAM contribution was obtained.",
        "These maps do not show contributions supporting the NORMAL class.",
        "Grad-CAM does not verify pneumonia localization or clinical validity.",
        "",
        "The saved evaluation predictions were not modified.",
        "The model was not retrained.",
        "The test set was not used.",
    ])

    (output / "gradcam_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(f"\nLargest score difference: {largest_difference:.8f}")
    print(f"Saved results to: {output}")


if __name__ == "__main__":
    main()