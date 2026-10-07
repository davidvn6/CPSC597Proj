"""
border_test_initial.py

Compare original validation predictions with border-masked predictions.

Run:
    .\\.venv\\Scripts\\python.exe .\\border_test_initial.py

Outputs inside the selected training run:
    border_test_initial/
        comparison_metrics.csv
        prediction_comparison.csv
        largest_score_changes.csv
        masking_examples.png
        experiment_config.json
        experiment_summary.txt

The model is not retrained.
Original images are not modified.
The test set is not used.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
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

# Reuse the exact architecture and image loader from training.
from train_initial import InitialCNN, XRayDataset


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "initial"

# Mask 5% of the resized image width/height on each side.
BORDER_FRACTION = 0.05

# Test two replacement values to examine sensitivity to masking choices.
CONDITIONS = ["original", "border_black", "border_mean"]


def mask_border(images, condition):
    """Replace outer border pixels without changing image dimensions."""

    if condition == "original":
        return images

    height = images.shape[-2]
    width = images.shape[-1]

    border_height = max(1, round(height * BORDER_FRACTION))
    border_width = max(1, round(width * BORDER_FRACTION))

    # True pixels identify the outer rectangular border.
    mask = torch.zeros(
        (1, 1, height, width),
        dtype=torch.bool,
        device=images.device,
    )
    mask[:, :, :border_height, :] = True
    mask[:, :, -border_height:, :] = True
    mask[:, :, :, :border_width] = True
    mask[:, :, :, -border_width:] = True

    if condition == "border_black":
        fill = torch.zeros_like(images[:, :, :1, :1])

    elif condition == "border_mean":
        # Calculate a separate mean intensity for each original image.
        fill = images.mean(dim=(2, 3), keepdim=True)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return torch.where(mask, fill, images)


def calculate_metrics(targets, scores, threshold):
    """Calculate validation metrics using pneumonia as the positive class."""

    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
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


def save_masking_examples(dataset, output_path):
    """Show the original and masked inputs for three fixed examples."""

    count = min(3, len(dataset))
    fig, axes = plt.subplots(
        count,
        len(CONDITIONS),
        figsize=(10, 3 * count),
        squeeze=False,
    )

    for row_index in range(count):
        image, target = dataset[row_index]
        batch = image.unsqueeze(0)

        for column, condition in enumerate(CONDITIONS):
            altered = mask_border(batch, condition)[0, 0].numpy()

            axes[row_index, column].imshow(
                altered,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(
                f"{condition}\n"
                f"Actual: {'PNEUMONIA' if target.item() == 1 else 'NORMAL'}"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    # Select the latest timestamped run containing a saved model.
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

    # Use the validation inventory saved with this experiment.
    val_path = run_folder / "val.csv"
    actual_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()
    expected_hash = config["split_file_sha256"]["val.csv"]

    if actual_hash != expected_hash:
        raise SystemExit("The validation CSV has changed since training.")

    table = pd.read_csv(val_path).reset_index(drop=True)

    if table.empty or set(table["target"]) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    # Match the original validation preprocessing.
    size = config["image_size"]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = XRayDataset(table, transform)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = InitialCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Comparing three conditions on {len(table)} validation images.",
        flush=True,
    )

    # Collect scores for all three versions of the same images.
    all_scores = {condition: [] for condition in CONDITIONS}

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            for condition in CONDITIONS:
                altered = mask_border(images, condition)
                logits = model(altered)
                scores = torch.sigmoid(logits)

                all_scores[condition].extend(
                    scores.cpu().tolist()
                )

    for condition in CONDITIONS:
        all_scores[condition] = np.asarray(
            all_scores[condition]
        )

    targets = table["target"].to_numpy(dtype=int)
    threshold = config["decision_threshold"]
    original_scores = all_scores["original"]
    original_predictions = (
        original_scores >= threshold
    ).astype(int)

    # Compare the newly calculated original scores against the earlier
    # evaluation, when that prediction file is available.
    previous_path = (
        run_folder / "validation_evaluation" / "predictions.csv"
    )

    if previous_path.is_file():
        previous = pd.read_csv(previous_path)

        if previous["path"].duplicated().any():
            raise SystemExit("Earlier predictions contain repeated paths.")

        if set(previous["path"]) != set(table["path"]):
            raise SystemExit("Earlier predictions use different images.")

        previous = previous.set_index("path").loc[
            table["path"]
        ]
        previous_scores = previous[
            "pneumonia_score"
        ].to_numpy()

        maximum_difference = float(
            np.max(np.abs(original_scores - previous_scores))
        )

        if maximum_difference > 0.001:
            raise SystemExit(
                "Original predictions differ from the earlier evaluation.\n"
                f"Maximum score difference: {maximum_difference:.8f}"
            )

    # Calculate performance and prediction changes for each condition.
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

        # Separate newly introduced errors from corrected errors.
        metrics["new_errors"] = int(np.sum(
            (original_predictions == targets)
            & (predictions != targets)
        ))
        metrics["corrected_errors"] = int(np.sum(
            (original_predictions != targets)
            & (predictions == targets)
        ))

        metric_rows.append(metrics)

        # Keep all per-image scores and predictions for paired comparisons.
        table[f"{condition}_score"] = scores
        table[f"{condition}_prediction"] = predictions
        table[f"{condition}_score_change"] = (
            scores - original_scores
        )

    metrics_table = pd.DataFrame(metric_rows)

    # Rank images by their largest score change under either masking method.
    table["largest_absolute_score_change"] = np.maximum(
        np.abs(all_scores["border_black"] - original_scores),
        np.abs(all_scores["border_mean"] - original_scores),
    )

    largest_changes = table.sort_values(
        "largest_absolute_score_change",
        ascending=False,
    ).head(20)

    out = run_folder / "border_test_initial"
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

    # Save example inputs to make the perturbation visible.
    save_masking_examples(
        dataset,
        out / "masking_examples.png",
    )

    border_pixels = max(1, round(size * BORDER_FRACTION))
    masked_fraction = 1 - (
        (size - 2 * border_pixels) / size
    ) ** 2

    experiment_config = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_images": len(table),
        "threshold": float(threshold),
        "image_size": size,
        "border_fraction_per_side": BORDER_FRACTION,
        "border_pixels_per_side": border_pixels,
        "fraction_of_image_pixels_masked": masked_fraction,
        "conditions": CONDITIONS,
        "mean_fill": "mean intensity of each original resized image",
        "mask_applied": "after resizing and scaling to [0,1]",
        "test_set_used": False,
        "model_retrained": False,
    }

    (out / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2),
        encoding="utf-8",
    )

    # Print the most relevant comparisons in a readable table.
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
        f"Border width: {border_pixels} pixels on each side\n"
        f"Image area masked: {100 * masked_fraction:.2f}%\n\n"
        f"{metrics_table[display_columns].round(4).to_string(index=False)}"
        "\n\n"
        "This is a validation sensitivity experiment.\n"
        "Masking may remove anatomy and introduce artificial boundaries.\n"
        "It does not isolate text markers or prove shortcut learning.\n"
        "Original images, model weights, and split assignments are unchanged.\n"
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