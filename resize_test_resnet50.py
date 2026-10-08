"""
resize_test_resnet50.py

Evaluate the saved ResNet-50 checkpoint under three input conditions:

    square_resize:
        Resize directly to a square, matching training preprocessing.

    aspect_black:
        Preserve aspect ratio and center the resized image on black padding.

    aspect_mean:
        Preserve aspect ratio and center the resized image on padding
        filled with the unpadded resized image's mean intensity.

Padding is applied BEFORE ImageNet normalization.

Run:
    .\\.venv\\Scripts\\python.exe .\\resize_test_resnet50.py

Outputs:
    results/resnet50/<training_run>/resize_test_resnet50/

No retraining is performed. The test set is not used.
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
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from train_initial import XRayDataset
from train_resnet50 import ResNet50CNN
from gradcam_resnet50 import get_validation_hash, sha256_file
from border_test_resnet50 import calculate_metrics


# ------------------------------------------------------------
# Project settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "resnet50"

SEED = 42
SCORE_TOLERANCE = 0.005

CONDITIONS = [
    "square_resize",
    "aspect_black",
    "aspect_mean",
]


class AspectPreservingResize:
    """Resize proportionally, then add centered padding to form a square."""

    def __init__(self, size, padding_mode):
        self.size = int(size)
        self.padding_mode = padding_mode

        if padding_mode not in {"black", "mean"}:
            raise ValueError("padding_mode must be 'black' or 'mean'.")

    def __call__(self, image):
        # Convert the grayscale image to a tensor before resizing.
        image = TF.to_image(image)
        height, width = image.shape[-2:]

        # Scale the longest side to the requested input size.
        scale = self.size / max(height, width)

        new_height = max(1, min(self.size, round(height * scale)))
        new_width = max(1, min(self.size, round(width * scale)))

        resized = TF.resize(
            image,
            size=[new_height, new_width],
            antialias=True,
        )
        resized = TF.to_dtype(
            resized,
            dtype=torch.float32,
            scale=True,
        )

        # Compute symmetric padding. An odd extra pixel goes right/bottom.
        left = (self.size - new_width) // 2
        right = self.size - new_width - left

        top = (self.size - new_height) // 2
        bottom = self.size - new_height - top

        if self.padding_mode == "black":
            fill = 0.0
        else:
            # Use the mean BEFORE padding, matching the InitialCNN experiment.
            fill = float(resized.mean())

        return TF.pad(
            resized,
            padding=[left, top, right, bottom],
            fill=fill,
            padding_mode="constant",
        )


def select_training_run():
    """Choose the latest ResNet-50 run with a saved best checkpoint."""
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit("No ResNet-50 checkpoint found.")

    return checkpoints[-1].parent


def make_transform(condition, image_size, mean, std):
    """Build the requested resize operation and training normalization."""
    if condition == "square_resize":
        resize_operations = [
            v2.ToImage(),
            v2.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            v2.ToDtype(torch.float32, scale=True),
        ]

    elif condition == "aspect_black":
        resize_operations = [
            AspectPreservingResize(image_size, "black"),
        ]

    elif condition == "aspect_mean":
        resize_operations = [
            AspectPreservingResize(image_size, "mean"),
        ]

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return v2.Compose([
        v2.Grayscale(num_output_channels=3),
        *resize_operations,
        v2.Normalize(mean=mean, std=std),
    ])


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
    # Load the checkpoint and verify validation assignments
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
    prediction_path = run / "validation_evaluation" / "predictions.csv"

    if not validation_path.exists():
        raise SystemExit(f"Missing validation CSV:\n{validation_path}")

    if not prediction_path.exists():
        raise SystemExit("Run evalmetric_resnet50.py first.")

    if sha256_file(validation_path) != get_validation_hash(config):
        raise SystemExit(
            "Validation assignments do not match the checkpoint configuration."
        )

    validation = pd.read_csv(validation_path)
    saved_predictions = pd.read_csv(prediction_path)

    if not validation["path"].is_unique:
        raise SystemExit("Duplicate paths in validation assignments.")

    if not saved_predictions["path"].is_unique:
        raise SystemExit("Duplicate paths in saved predictions.")

    if set(validation["path"]) != set(saved_predictions["path"]):
        raise SystemExit(
            "Saved predictions do not match the validation images."
        )

    # Align saved predictions with the validation CSV's row order.
    saved_predictions = (
        saved_predictions.set_index("path")
        .loc[validation["path"]]
        .reset_index()
    )

    targets = validation["target"].to_numpy(dtype=int)

    if set(targets) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    if not np.array_equal(
        targets,
        saved_predictions["target"].to_numpy(dtype=int),
    ):
        raise SystemExit("Validation labels differ from saved predictions.")

    # --------------------------------------------------------
    # Load model and recorded preprocessing settings
    # --------------------------------------------------------

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    threshold = float(config["decision_threshold"])

    mean = config["normalization_mean"]
    std = config["normalization_std"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load all weights from the checkpoint without another download.
    model = ResNet50CNN(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores_by_condition = {}

    # --------------------------------------------------------
    # Evaluate each resize condition
    # --------------------------------------------------------

    for condition in CONDITIONS:
        print(f"Evaluating {condition}...")

        transform = make_transform(
            condition,
            image_size,
            mean,
            std,
        )

        dataset = XRayDataset(validation, transform)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        batches = []

        # Use full precision, matching the validation evaluation.
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device, dtype=torch.float32)

                logits = model(images).reshape(-1)
                scores = torch.sigmoid(logits).cpu().numpy()

                batches.append(scores)

        combined = np.concatenate(batches)

        if len(combined) != len(validation):
            raise SystemExit(f"Wrong prediction count for {condition}.")

        if not np.isfinite(combined).all():
            raise SystemExit(f"Non-finite scores for {condition}.")

        scores_by_condition[condition] = combined

        # Verify the reference condition before evaluating alternatives.
        if condition == "square_resize":
            saved_scores = saved_predictions["pneumonia_score"].to_numpy()

            largest_difference = float(
                np.max(np.abs(combined - saved_scores))
            )

            if largest_difference > SCORE_TOLERANCE:
                raise SystemExit(
                    "Square-resize scores differ from the saved evaluation.\n"
                    f"Largest difference: {largest_difference:.8f}\n"
                    "Check the checkpoint and preprocessing."
                )

            predicted = (combined >= threshold).astype(int)

            if not np.array_equal(
                predicted,
                saved_predictions["predicted_target"].to_numpy(dtype=int),
            ):
                raise SystemExit(
                    "Square-resize classifications differ from saved predictions.\n"
                    "Resolve this before interpreting the resize experiment."
                )

    # --------------------------------------------------------
    # Calculate metrics and paired prediction changes
    # --------------------------------------------------------

    reference_scores = scores_by_condition["square_resize"]
    reference_predictions = (
        reference_scores >= threshold
    ).astype(int)

    paired = validation.copy()
    metric_rows = []

    for condition in CONDITIONS:
        scores = scores_by_condition[condition]
        predicted = (scores >= threshold).astype(int)

        # Reuse the same metric calculations as the border experiment.
        metrics = calculate_metrics(
            targets,
            scores,
            threshold,
            reference_scores,
        )
        metric_rows.append({"condition": condition, **metrics})

        paired[f"{condition}_score"] = scores
        paired[f"{condition}_prediction"] = predicted
        paired[f"{condition}_correct"] = predicted == targets
        paired[f"{condition}_flipped"] = (
            predicted != reference_predictions
        )

    metrics_table = pd.DataFrame(metric_rows)

    # --------------------------------------------------------
    # Save results and experiment settings
    # --------------------------------------------------------

    output = run / "resize_test_resnet50"
    output.mkdir(parents=True, exist_ok=True)

    metrics_table.to_csv(output / "metrics.csv", index=False)
    paired.to_csv(output / "paired_predictions.csv", index=False)

    changed = (
        paired["aspect_black_flipped"]
        | paired["aspect_mean_flipped"]
    )

    paired.loc[changed].to_csv(
        output / "changed_predictions.csv",
        index=False,
    )

    settings = {
        "training_run": str(run),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_images": len(validation),
        "validation_sha256": sha256_file(validation_path),
        "image_size": image_size,
        "batch_size": batch_size,
        "threshold": threshold,
        "conditions": CONDITIONS,
        "padding_position": "centered",
        "mean_padding_source": "unpadded resized image",
        "padding_before_normalization": True,
        "largest_reference_score_difference": largest_difference,
    }

    (output / "experiment_config.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save a comparison plot for the report
    # --------------------------------------------------------

    plot_table = metrics_table.set_index("condition")[
        ["accuracy", "recall", "specificity", "roc_auc"]
    ]

    ax = plot_table.plot(
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )
    ax.set_title("ResNet-50 validation resize sensitivity")
    ax.set_xlabel("Input condition")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output / "metric_comparison.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Print and save the experiment summary
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
        "",
        displayed,
        "",
        "The model was trained with square-resized images.",
        "Aspect-preserving resize changes geometry and introduces padding.",
        "Performance changes indicate preprocessing sensitivity.",
        "They do not prove reliance on aspect-ratio distortion.",
        "This experiment alone does not establish better training preprocessing.",
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