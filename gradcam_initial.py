"""
gradcam_initial.py

Generate Grad-CAM heatmaps for the saved initial CNN.

Run:
    .\\.venv\\Scripts\\python.exe .\\gradcam_initial.py

Outputs inside the selected training run:
    gradcam_initial/
        selected_examples.csv
        gradcam_summary.txt
        PNG figures and NumPy heatmap arrays

All heatmaps target the PNEUMONIA score, including NORMAL predictions.
The script does not retrain the model or use the test set.
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

# Reuse the architecture that produced the saved checkpoint.
from train_initial import InitialCNN


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "initial"

# Select a small, reproducible sample from each prediction category.
SEED = 42
EXAMPLES_PER_CATEGORY = 3
CATEGORIES = ["TP", "TN", "FP", "FN"]


def generate_gradcam(model, image_tensor):
    """
    Generate a map of positive contributions to the PNEUMONIA logit.

    Use the activated output of the final convolution block,
    before its final max-pooling operation.
    """

    captured = {}

    def capture_features(module, inputs, output):
        captured["features"] = output

    # InitialCNN has four Conv2d/ReLU/MaxPool blocks.
    # Index 10 is the ReLU after the final convolution at index 9.
    handle = model.features[10].register_forward_hook(
        capture_features
    )

    try:
        # Grad-CAM requires gradients even though the model is in eval mode.
        with torch.enable_grad():
            logits = model(image_tensor)
            features = captured["features"]

            # Differentiate the pneumonia logit rather than its sigmoid score.
            gradients = torch.autograd.grad(
                outputs=logits[0],
                inputs=features,
            )[0]

            # Average gradients across spatial positions to weight channels.
            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            # Combine feature channels and retain positive contributions.
            raw_map = (weights * features).sum(
                dim=1,
                keepdim=True,
            )
            raw_map = torch.relu(raw_map)

            raw_max = float(raw_map.max().detach().cpu())

            # Enlarge the coarse map to the model's input resolution.
            heatmap = F.interpolate(
                raw_map,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Normalize each image's map independently for visualization.
            # A zero map means no positive contribution survived the ReLU.
            maximum = heatmap.max()

            if float(maximum.detach().cpu()) > 0:
                heatmap = heatmap / maximum

            score = float(
                torch.sigmoid(logits[0]).detach().cpu()
            )

            return (
                heatmap[0, 0].detach().cpu().numpy(),
                score,
                raw_max,
            )

    finally:
        # Remove the hook so it does not accumulate between images.
        handle.remove()


def save_figure(input_image, heatmap, row, score, output_path):
    """Save the model input, heatmap, and overlay side by side."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(
        input_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Model input")

    # Use a fixed displayed scale after per-image normalization.
    heatmap_plot = axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("PNEUMONIA Grad-CAM")

    axes[2].imshow(
        input_image,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # Make zero-valued heatmap regions transparent.
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

    fig.colorbar(
        heatmap_plot,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Relative positive contribution",
    )

    fig.suptitle(
        f"{row['outcome']} | Actual: {row['label']} | "
        f"Predicted: {row['predicted_label']}\n"
        f"Pneumonia score: {score:.4f} | "
        f"{Path(row['path']).name}",
        fontsize=10,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.87))
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    # Select the latest timestamped run containing a saved model.
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit(f"No checkpoint found inside: {RUNS}")

    checkpoint_path = checkpoints[-1]
    run_folder = checkpoint_path.parent

    print(f"Selected training run: {run_folder}", flush=True)

    predictions_path = (
        run_folder / "validation_evaluation" / "predictions.csv"
    )

    if not predictions_path.is_file():
        raise SystemExit(
            "Run evalmetric_initial.py for this training run first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    if config["architecture"] != "InitialCNN: channels 16,32,64,128":
        raise SystemExit("The checkpoint does not match InitialCNN.")

    # Verify the validation file has not changed since training.
    val_path = run_folder / "val.csv"
    actual_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()
    expected_hash = config["split_file_sha256"]["val.csv"]

    if actual_hash != expected_hash:
        raise SystemExit("The saved validation CSV has changed.")

    validation = pd.read_csv(val_path)
    predictions = pd.read_csv(predictions_path)

    # Check that predictions cover the saved validation inventory.
    if predictions["path"].duplicated().any():
        raise SystemExit("The prediction file contains repeated paths.")

    if set(predictions["path"]) != set(validation["path"]):
        raise SystemExit(
            "Predictions do not match the saved validation inventory."
        )

    if not predictions["outcome"].isin(CATEGORIES).all():
        raise SystemExit("Unexpected prediction categories found.")

    # Sample from each category without choosing only extreme errors.
    selections = []

    for category in CATEGORIES:
        candidates = predictions[
            predictions["outcome"] == category
        ].sort_values("path")

        if not candidates.empty:
            sample = candidates.sample(
                n=min(EXAMPLES_PER_CATEGORY, len(candidates)),
                random_state=SEED,
            )
            selections.append(sample)

    if not selections:
        raise SystemExit("No prediction examples were found.")

    selected = pd.concat(selections, ignore_index=True)

    # Recreate the deterministic validation preprocessing.
    size = config["image_size"]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = InitialCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Disable dropout while retaining gradient computation for Grad-CAM.
    model.eval()

    out = run_folder / "gradcam_initial"
    out.mkdir(parents=True, exist_ok=True)

    records = []

    for index, row in selected.iterrows():
        # Load and transform the image exactly as during validation.
        with Image.open(ROOT / row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image_tensor = transform(image)

        input_image = image_tensor[0].numpy()
        batch = image_tensor.unsqueeze(0).to(device)

        heatmap, score, raw_max = generate_gradcam(model, batch)

        # Check that this checkpoint reproduces the saved prediction score.
        # Compare single-image inference against the saved batch prediction.
        # Allow a small numerical difference, but stop for larger differences.
        saved_score = float(row["pneumonia_score"])
        difference = abs(score - saved_score)

        if difference > 0.001:
            raise SystemExit(
                f"Score mismatch for {row['path']}\n"
                f"Saved batch score: {saved_score:.8f}\n"
                f"Grad-CAM score:    {score:.8f}\n"
                f"Absolute difference: {difference:.8f}\n"
                "Send these values so we can investigate."
            )

        if difference > 0.0001:
            print(
                f"Small score difference for {Path(row['path']).name}: "
                f"{difference:.8f}",
                flush=True,
            )

        stem = Path(row["path"]).stem
        filename = f"{index + 1:02d}_{row['outcome']}_{stem}"

        # Save the normalized numeric map and its visualization.
        np.save(out / f"{filename}.npy", heatmap)

        save_figure(
            input_image,
            heatmap,
            row,
            score,
            out / f"{filename}.png",
        )

        record = row.to_dict()
        record["gradcam_target"] = "PNEUMONIA"
        record["recomputed_score"] = score
        record["raw_positive_map_max"] = raw_max
        record["zero_positive_map"] = raw_max == 0
        record["figure"] = f"{filename}.png"
        records.append(record)

        print(
            f"Saved {row['outcome']}: {stem}",
            flush=True,
        )

    # Save the selected examples and method details for the report.
    pd.DataFrame(records).to_csv(
        out / "selected_examples.csv",
        index=False,
    )

    summary = (
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Examples generated: {len(records)}\n"
        f"Sampling seed: {SEED}\n"
        "Source split: validation\n"
        "Target score: PNEUMONIA logit for every example\n"
        "Target features: final convolution's ReLU output, "
        "before final max pooling\n"
        "Maps are normalized separately for each image.\n"
        "Heatmap intensity is not comparable across images.\n"
        "Zero maps indicate no positive contribution at this target layer.\n"
        "Figures display resized model inputs, not original-resolution images.\n"
        "Heatmaps do not verify clinical disease localization.\n"
        "The model was not retrained and the test set was not used.\n"
    )

    (out / "gradcam_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(f"\nSaved Grad-CAM results to: {out}")


if __name__ == "__main__":
    main()