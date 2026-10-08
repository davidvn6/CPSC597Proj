"""
resize_test_initial.py

Test sensitivity to aspect-preserving image resizing.

Run:
    .\\.venv\\Scripts\\python.exe .\\resize_test_initial.py

Outputs inside the selected training run:
    resize_test_initial/
        comparison_metrics.csv
        prediction_comparison.csv
        largest_score_changes.csv
        resize_examples.png
        experiment_config.json
        experiment_summary.txt

The model is not retrained.
Original images and split assignments are unchanged.
The test set is not used.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Reuse the model and image loader from the original training script.
from train_initial import InitialCNN, XRayDataset


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "initial"

CONDITIONS = [
    "square_resize",
    "aspect_black",
    "aspect_mean",
]


class AspectPreservingResize:
    """Resize to fit a square and pad without stretching the image."""

    def __init__(self, size, padding_method):
        self.size = size
        self.padding_method = padding_method
        self.to_image = v2.ToImage()
        self.to_float = v2.ToDtype(
            torch.float32,
            scale=True,
        )

    def __call__(self, image):
        # Convert the grayscale PIL image to a tensor.
        image = self.to_image(image)
        height, width = image.shape[-2:]

        # Scale both dimensions equally so the longer side fits the square.
        scale = self.size / max(height, width)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))

        image = v2.Resize(
            (new_height, new_width),
            antialias=True,
        )(image)

        # Scale pixels to [0, 1] before choosing the padding intensity.
        image = self.to_float(image)

        if self.padding_method == "black":
            fill_value = 0.0
        elif self.padding_method == "mean":
            fill_value = float(image.mean())
        else:
            raise ValueError("Unknown padding method.")

        # Center the resized image within the target square.
        remaining_height = self.size - new_height
        remaining_width = self.size - new_width

        top = remaining_height // 2
        bottom = remaining_height - top
        left = remaining_width // 2
        right = remaining_width - left

        # PyTorch padding order is left, right, top, bottom.
        image = F.pad(
            image,
            (left, right, top, bottom),
            mode="constant",
            value=fill_value,
        )

        return image


def calculate_metrics(targets, scores, threshold):
    """Calculate metrics with pneumonia as the positive class."""

    predictions = (scores >= threshold).astype(int)

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(
            targets, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            targets, predictions, zero_division=0
        )),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(
            targets, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_resize_examples(table, datasets, output_path):
    """Show the three input versions for the same selected images."""

    # Select different aspect ratios to make the resize differences visible.
    aspect_ratios = table["width"] / table["height"]
    ordered_indices = aspect_ratios.sort_values().index.tolist()

    positions = [
        0,
        len(ordered_indices) // 2,
        len(ordered_indices) - 1,
    ]

    # Avoid repeating examples if the table is unusually small.
    selected_indices = []

    for position in positions:
        index = ordered_indices[position]

        if index not in selected_indices:
            selected_indices.append(index)

    fig, axes = plt.subplots(
        len(selected_indices),
        len(CONDITIONS),
        figsize=(11, 3.5 * len(selected_indices)),
        squeeze=False,
    )

    for row_number, index in enumerate(selected_indices):
        row = table.iloc[index]
        ratio = float(aspect_ratios.iloc[index])

        for column, condition in enumerate(CONDITIONS):
            image, target = datasets[condition][index]

            axes[row_number, column].imshow(
                image[0].numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row_number, column].axis("off")
            axes[row_number, column].set_title(
                f"{condition}\n"
                f"Actual: {row['label']} | "
                f"Original width/height: {ratio:.2f}"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    # Select the latest timestamped run containing a saved checkpoint.
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit(f"No saved model found inside: {RUNS}")

    checkpoint_path = checkpoints[-1]
    run_folder = checkpoint_path.parent

    print(f"Selected training run: {run_folder}", flush=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    if config["architecture"] != "InitialCNN: channels 16,32,64,128":
        raise SystemExit("The checkpoint does not match InitialCNN.")

    # Verify the validation inventory saved with the training run.
    val_path = run_folder / "val.csv"

    actual_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()
    expected_hash = config["split_file_sha256"]["val.csv"]

    if actual_hash != expected_hash:
        raise SystemExit("The validation CSV has changed since training.")

    table = pd.read_csv(val_path).reset_index(drop=True)

    if table.empty or set(table["target"]) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    # Reproduce the original square-resize preprocessing exactly.
    size = config["image_size"]

    transforms = {
        "square_resize": v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]),
        "aspect_black": AspectPreservingResize(
            size,
            padding_method="black",
        ),
        "aspect_mean": AspectPreservingResize(
            size,
            padding_method="mean",
        ),
    }

    datasets = {}

    for condition in CONDITIONS:
        datasets[condition] = XRayDataset(
            table,
            transforms[condition],
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = InitialCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    targets = table["target"].to_numpy(dtype=int)
    threshold = config["decision_threshold"]

    all_scores = {}

    # Evaluate each preprocessing condition in the same CSV order.
    for condition in CONDITIONS:
        print(f"Evaluating {condition}...", flush=True)

        loader = DataLoader(
            datasets[condition],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        scores = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                batch_scores = torch.sigmoid(logits)
                scores.extend(batch_scores.cpu().tolist())

        all_scores[condition] = np.asarray(scores)

    original_scores = all_scores["square_resize"]
    original_predictions = (
        original_scores >= threshold
    ).astype(int)

    # Confirm the control condition agrees with the earlier evaluation.
    previous_path = (
        run_folder / "validation_evaluation" / "predictions.csv"
    )

    if not previous_path.is_file():
        raise SystemExit("Run evalmetric_initial.py first.")

    previous = pd.read_csv(previous_path)

    if previous["path"].duplicated().any():
        raise SystemExit("Earlier predictions contain repeated paths.")

    if set(previous["path"]) != set(table["path"]):
        raise SystemExit("Earlier predictions use different images.")

    previous = previous.set_index("path").loc[table["path"]]
    previous_scores = previous["pneumonia_score"].to_numpy()

    maximum_difference = float(
        np.max(np.abs(original_scores - previous_scores))
    )

    if maximum_difference > 0.001:
        raise SystemExit(
            "Square-resize scores differ from the earlier evaluation.\n"
            f"Maximum difference: {maximum_difference:.8f}"
        )

    # Compare metrics and prediction changes against square resizing.
    metric_rows = []

    for condition in CONDITIONS:
        scores = all_scores[condition]
        predictions = (scores >= threshold).astype(int)

        metrics = calculate_metrics(
            targets,
            scores,
            threshold,
        )
        metrics["condition"] = condition
        metrics["mean_score_change"] = float(
            np.mean(scores - original_scores)
        )
        metrics["mean_absolute_score_change"] = float(
            np.mean(np.abs(scores - original_scores))
        )
        metrics["prediction_flips"] = int(
            np.sum(predictions != original_predictions)
        )
        metrics["new_errors"] = int(np.sum(
            (original_predictions == targets)
            & (predictions != targets)
        ))
        metrics["corrected_errors"] = int(np.sum(
            (original_predictions != targets)
            & (predictions == targets)
        ))

        metric_rows.append(metrics)

        # Save per-image scores for paired comparisons.
        table[f"{condition}_score"] = scores
        table[f"{condition}_prediction"] = predictions
        table[f"{condition}_score_change"] = (
            scores - original_scores
        )

    metrics_table = pd.DataFrame(metric_rows)

    # Find images most affected by either aspect-preserving method.
    table["largest_absolute_score_change"] = np.maximum(
        np.abs(all_scores["aspect_black"] - original_scores),
        np.abs(all_scores["aspect_mean"] - original_scores),
    )

    largest_changes = table.sort_values(
        "largest_absolute_score_change",
        ascending=False,
    ).head(20)

    out = run_folder / "resize_test_initial"
    out.mkdir(parents=True, exist_ok=True)

    metrics_table.to_csv(
        out / "comparison_metrics.csv",
        index=False,
    )
    table.to_csv(
        out / "prediction_comparison.csv",
        index=False,
    )
    largest_changes.to_csv(
        out / "largest_score_changes.csv",
        index=False,
    )

    save_resize_examples(
        table,
        datasets,
        out / "resize_examples.png",
    )

    # Save the experiment settings for the report.
    experiment_config = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_images": len(table),
        "image_size": size,
        "threshold": float(threshold),
        "conditions": CONDITIONS,
        "aspect_resize": "scale longer side to image_size",
        "padding_position": "centered",
        "mean_padding": "mean of the resized, unpadded image",
        "pixel_range": [0, 1],
        "maximum_control_score_difference": maximum_difference,
        "model_retrained": False,
        "test_set_used": False,
    }

    (out / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "condition",
        "accuracy",
        "recall",
        "specificity",
        "roc_auc",
        "false_negatives",
        "false_positives",
        "prediction_flips",
        "mean_absolute_score_change",
    ]

    summary = (
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Validation images: {len(table)}\n"
        f"Decision threshold: {threshold}\n\n"
        f"{metrics_table[display_columns].round(4).to_string(index=False)}"
        "\n\n"
        "The model was trained with square-resized images.\n"
        "Aspect-preserving resize changes geometry and introduces padding.\n"
        "Performance changes indicate preprocessing sensitivity, "
        "not proof of reliance on distortion.\n"
        "This experiment alone does not select the next model's preprocessing.\n"
        "Original images, weights, and split assignments are unchanged.\n"
        "The test set was not used.\n"
    )

    (out / "experiment_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()