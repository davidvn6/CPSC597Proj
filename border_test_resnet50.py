"""
border_test_resnet50.py

Compare three conditions on the saved validation set:
    original      Unchanged square-resized input.
    border_black  Outer border replaced with black.
    border_mean   Outer border replaced with the resized image's mean.

Masking happens BEFORE ImageNet normalization.

Run:
    .\\.venv\\Scripts\\python.exe .\\border_test_resnet50.py

Outputs:
    results/resnet50/<training_run>/border_test_resnet50/

This script does not retrain the model or evaluate the test set.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from train_initial import XRayDataset
from train_resnet50 import ResNet50CNN
from gradcam_resnet50 import get_validation_hash, sha256_file


# ------------------------------------------------------------
# Project settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "resnet50"

SEED = 42
BORDER_FRACTION = 0.05
CONDITIONS = ["original", "border_black", "border_mean"]

# Check that original predictions reproduce the saved evaluation.
SCORE_TOLERANCE = 0.005


def select_training_run():
    """Find the latest ResNet-50 run containing a best checkpoint."""
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit("No ResNet-50 checkpoint found.")

    return checkpoints[-1].parent


def mask_border(images, condition, border):
    """
    Replace the outer border of a batch of unnormalized images.

    images has shape:
        batch size x channels x height x width
    """
    if condition == "original":
        return images

    masked = images.clone()

    if condition == "border_black":
        fill = torch.zeros_like(images[:, :, :1, :1])

    elif condition == "border_mean":
        # Calculate each image's mean from the complete resized input.
        fill = images.mean(dim=(2, 3), keepdim=True)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Broadcast each image's fill value into all four border strips.
    masked[:, :, :border, :] = fill
    masked[:, :, -border:, :] = fill
    masked[:, :, :, :border] = fill
    masked[:, :, :, -border:] = fill

    return masked


def calculate_metrics(targets, scores, threshold, original_scores):
    """Calculate classification metrics and paired prediction changes."""
    predicted = (scores >= threshold).astype(int)
    original_predicted = (original_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predicted,
        labels=[0, 1],
    ).ravel()

    original_correct = original_predicted == targets
    current_correct = predicted == targets

    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "precision": float(
            precision_score(targets, predicted, zero_division=0)
        ),
        "recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(targets, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "prediction_flips": int(
            np.sum(predicted != original_predicted)
        ),
        "new_errors": int(
            np.sum(original_correct & ~current_correct)
        ),
        "corrected_errors": int(
            np.sum(~original_correct & current_correct)
        ),
        "mean_absolute_score_change": float(
            np.mean(np.abs(scores - original_scores))
        ),
    }


def main():
    # --------------------------------------------------------
    # Set reproducibility options
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------------------------------------
    # Load the checkpoint and saved validation assignments
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
        raise SystemExit("The checkpoint is not a ResNet50CNN model.")

    validation_path = run / "val.csv"
    saved_prediction_path = (
        run / "validation_evaluation" / "predictions.csv"
    )

    if not validation_path.exists():
        raise SystemExit(f"Missing validation CSV: {validation_path}")

    if not saved_prediction_path.exists():
        raise SystemExit("Run evalmetric_resnet50.py first.")

    if sha256_file(validation_path) != get_validation_hash(config):
        raise SystemExit(
            "Validation assignments do not match the checkpoint configuration."
        )

    validation = pd.read_csv(validation_path)
    saved_predictions = pd.read_csv(saved_prediction_path)

    if not validation["path"].is_unique:
        raise SystemExit("Duplicate paths in the validation CSV.")

    if not saved_predictions["path"].is_unique:
        raise SystemExit("Duplicate paths in the saved predictions.")

    if set(validation["path"]) != set(saved_predictions["path"]):
        raise SystemExit(
            "Saved predictions do not match the validation images."
        )

    if set(validation["target"].astype(int)) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    # Align saved predictions with the validation CSV's row order.
    saved_predictions = (
        saved_predictions.set_index("path")
        .loc[validation["path"]]
        .reset_index()
    )

    targets = validation["target"].to_numpy(dtype=int)

    if not np.array_equal(
        targets,
        saved_predictions["target"].to_numpy(dtype=int),
    ):
        raise SystemExit("Validation labels differ from saved predictions.")

    # --------------------------------------------------------
    # Prepare preprocessing and model
    # --------------------------------------------------------

    image_size = int(config["image_size"])
    threshold = float(config["decision_threshold"])
    batch_size = int(config["batch_size"])

    border = max(1, round(image_size * BORDER_FRACTION))

    if 2 * border >= image_size:
        raise SystemExit("Border width leaves no unmasked center.")

    masked_fraction = (
        1 - ((image_size - 2 * border) / image_size) ** 2
    )

    # Return pixels in [0, 1]. Normalize only AFTER masking.
    transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.Resize(
            (image_size, image_size),
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = XRayDataset(validation, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use the normalization values recorded during training.
    mean = torch.tensor(
        config["normalization_mean"],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        config["normalization_std"],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    model = ResNet50CNN(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Comparing three conditions on {len(validation)} validation images.")

    # --------------------------------------------------------
    # Evaluate all conditions using the same batches
    # --------------------------------------------------------

    collected = {condition: [] for condition in CONDITIONS}

    # Full-precision evaluation; no training or gradient calculation.
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, dtype=torch.float32)

            for condition in CONDITIONS:
                altered = mask_border(images, condition, border)
                normalized = (altered - mean) / std

                logits = model(normalized).reshape(-1)
                scores = torch.sigmoid(logits).cpu().numpy()

                collected[condition].append(scores)

    scores_by_condition = {
        condition: np.concatenate(batches)
        for condition, batches in collected.items()
    }

    for condition, scores in scores_by_condition.items():
        if len(scores) != len(validation) or not np.isfinite(scores).all():
            raise SystemExit(f"Invalid scores for {condition}.")

    # --------------------------------------------------------
    # Verify reproduction of the original evaluation
    # --------------------------------------------------------

    original_scores = scores_by_condition["original"]
    saved_scores = saved_predictions["pneumonia_score"].to_numpy()

    largest_difference = float(
        np.max(np.abs(original_scores - saved_scores))
    )

    if largest_difference > SCORE_TOLERANCE:
        raise SystemExit(
            "Original scores differ from the saved evaluation.\n"
            f"Largest difference: {largest_difference:.8f}\n"
            "Check preprocessing and checkpoint consistency."
        )

    original_predicted = (original_scores >= threshold).astype(int)

    if not np.array_equal(
        original_predicted,
        saved_predictions["predicted_target"].to_numpy(dtype=int),
    ):
        raise SystemExit(
            "Original classifications differ from the saved evaluation.\n"
            "Resolve this difference before interpreting the sensitivity test."
        )

    # --------------------------------------------------------
    # Calculate metrics and save paired predictions
    # --------------------------------------------------------

    output = run / "border_test_resnet50"
    output.mkdir(parents=True, exist_ok=True)

    paired = validation.copy()
    metric_rows = []

    for condition in CONDITIONS:
        scores = scores_by_condition[condition]
        predicted = (scores >= threshold).astype(int)

        metrics = calculate_metrics(
            targets,
            scores,
            threshold,
            original_scores,
        )
        metric_rows.append({"condition": condition, **metrics})

        paired[f"{condition}_score"] = scores
        paired[f"{condition}_prediction"] = predicted
        paired[f"{condition}_correct"] = predicted == targets
        paired[f"{condition}_flipped"] = (
            predicted != original_predicted
        )

    metrics_table = pd.DataFrame(metric_rows)

    metrics_table.to_csv(output / "metrics.csv", index=False)
    paired.to_csv(output / "paired_predictions.csv", index=False)

    # Save the images whose classifications changed under either mask.
    changed = (
        paired["border_black_flipped"]
        | paired["border_mean_flipped"]
    )
    paired.loc[changed].to_csv(
        output / "changed_predictions.csv",
        index=False,
    )

    # Record the experiment settings.
    settings = {
        "training_run": str(run),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_images": len(validation),
        "validation_sha256": sha256_file(validation_path),
        "image_size": image_size,
        "batch_size": batch_size,
        "threshold": threshold,
        "border_fraction_requested": BORDER_FRACTION,
        "border_pixels": border,
        "masked_area_fraction": masked_fraction,
        "masking_before_normalization": True,
        "largest_original_score_difference": largest_difference,
    }

    (output / "experiment_config.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save a comparison figure for the report
    # --------------------------------------------------------

    plot_table = metrics_table.set_index("condition")[
        ["accuracy", "recall", "specificity", "roc_auc"]
    ]

    ax = plot_table.plot(
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )
    ax.set_title("ResNet-50 validation border sensitivity")
    ax.set_xlabel("Input condition")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output / "metric_comparison.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Print and save the summary
    # --------------------------------------------------------

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

    displayed = metrics_table[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )

    summary = "\n".join([
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Validation images: {len(validation)}",
        f"Decision threshold: {threshold}",
        f"Border width: {border} pixels on each side",
        f"Image area masked: {100 * masked_fraction:.2f}%",
        "",
        displayed,
        "",
        "This is a validation sensitivity experiment.",
        "Masking may remove anatomy and introduce artificial boundaries.",
        "It does not isolate text markers or prove shortcut learning.",
        "Scores are not calibrated clinical probabilities.",
        "Original images, weights, and split assignments were unchanged.",
        "The test set was not used.",
    ])

    (output / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()