"""
evalmetric_initial.py

Evaluate the saved initial CNN on its validation set.

Run:
    .\\.venv\\Scripts\\python.exe .\\evalmetric_initial.py

Outputs inside the selected training run's validation_evaluation folder:
    predictions.csv
    false_negatives.csv
    false_positives.csv
    metrics.json
    evaluation_summary.txt
    confusion_matrix.png
    roc_curve.png
    false_negative_examples.png
    false_positive_examples.png

This script does not retrain the model or use the test set.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Reuse the exact model and dataset classes from the training script.
# Importing this file does not start training because it has a main guard.
from train_initial import InitialCNN, XRayDataset


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "initial"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def save_error_examples(table, title, output_path):
    """Display up to six errors with the strongest incorrect scores."""

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for axis in axes.flat:
        axis.axis("off")

    for index in range(min(6, len(table))):
        row = table.iloc[index]
        axis = axes.flat[index]

        # Display the original grayscale image.
        with Image.open(ROOT / row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            axis.imshow(image, cmap="gray")

        axis.set_title(
            f"Actual: {row['label']}\n"
            f"Predicted: {row['predicted_label']}\n"
            f"Pneumonia score: {row['pneumonia_score']:.3f}"
        )

    if table.empty:
        fig.text(
            0.5,
            0.5,
            "No errors in this category.",
            ha="center",
        )

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    # Find saved checkpoints and select the latest timestamped run.
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit(f"No saved model found inside: {RUNS}")

    checkpoint_path = checkpoints[-1]
    run_folder = checkpoint_path.parent
    print(f"Selected training run: {run_folder}", flush=True)

    # Load the saved weights and experiment configuration onto the CPU first.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    # Stop if this checkpoint belongs to a different architecture.
    if config["architecture"] != "InitialCNN: channels 16,32,64,128":
        raise SystemExit("This checkpoint does not match InitialCNN.")

    # Use the validation CSV saved with this experiment.
    val_path = run_folder / "val.csv"

    # Confirm the validation file matches the recorded configuration.
    expected_hash = config["split_file_sha256"]["val.csv"]
    actual_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()

    if actual_hash != expected_hash:
        raise SystemExit(
            "The saved validation CSV has changed since training."
        )

    table = pd.read_csv(val_path).reset_index(drop=True)

    if table.empty or set(table["target"]) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    # Use the same deterministic preprocessing as training validation.
    image_size = config["image_size"]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(
            (image_size, image_size),
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = XRayDataset(table, transform)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Prefer the GPU but allow CPU evaluation if needed.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = InitialCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Disable dropout and use the model's evaluation behavior.
    model.eval()

    print(
        f"Evaluating epoch {checkpoint['epoch']} "
        f"on {len(table)} validation images using {device}.",
        flush=True,
    )

    # Predict in CSV order so scores align with the correct image paths.
    scores = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            batch_scores = torch.sigmoid(logits)
            scores.extend(batch_scores.cpu().tolist())

    scores = np.asarray(scores)
    targets = table["target"].to_numpy(dtype=int)

    # Retain the threshold recorded during training.
    threshold = config["decision_threshold"]
    predictions = (scores >= threshold).astype(int)

    # Record each image's score, predicted class, and correctness.
    table["pneumonia_score"] = scores
    table["predicted_target"] = predictions
    table["predicted_label"] = table["predicted_target"].map({
        0: "NORMAL",
        1: "PNEUMONIA",
    })
    table["correct"] = predictions == targets

    # Assign a confusion-matrix category to each prediction.
    outcomes = []

    for index in range(len(table)):
        actual = targets[index]
        predicted = predictions[index]

        if actual == 1 and predicted == 1:
            outcomes.append("TP")
        elif actual == 0 and predicted == 0:
            outcomes.append("TN")
        elif actual == 0 and predicted == 1:
            outcomes.append("FP")
        else:
            outcomes.append("FN")

    table["outcome"] = outcomes

    # Calculate metrics with pneumonia as the positive class.
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "split": "validation",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "images": len(table),
        "threshold": float(threshold),
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

    out = run_folder / "validation_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    # Save individual predictions and separate error inventories.
    table.to_csv(out / "predictions.csv", index=False)

    # Lowest pneumonia scores are the strongest false negatives.
    false_negatives = table[table["outcome"] == "FN"].sort_values(
        "pneumonia_score"
    )

    # Highest pneumonia scores are the strongest false positives.
    false_positives = table[table["outcome"] == "FP"].sort_values(
        "pneumonia_score",
        ascending=False,
    )

    false_negatives.to_csv(
        out / "false_negatives.csv",
        index=False,
    )
    false_positives.to_csv(
        out / "false_positives.csv",
        index=False,
    )

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Plot the confusion matrix with explicit class names.
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(
        f"Initial CNN: validation confusion matrix\n"
        f"Epoch {checkpoint['epoch']}, threshold {threshold}"
    )
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Plot the ROC curve using continuous prediction scores.
    fpr, tpr, thresholds = roc_curve(targets, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Initial CNN: validation ROC curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=200)
    plt.close(fig)

    # Save selected error examples for inspection.
    save_error_examples(
        false_negatives,
        "Validation false negatives: lowest pneumonia scores",
        out / "false_negative_examples.png",
    )
    save_error_examples(
        false_positives,
        "Validation false positives: highest pneumonia scores",
        out / "false_positive_examples.png",
    )

    # Produce a readable summary for the terminal and project report.
    report = classification_report(
        targets,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    summary = (
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Validation images: {len(table)}\n"
        f"Decision threshold: {threshold}\n\n"
        f"{report}\n"
        f"Specificity: {metrics['specificity']:.4f}\n"
        f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
        f"True negatives: {tn}\n"
        f"False positives: {fp}\n"
        f"False negatives: {fn}\n"
        f"True positives: {tp}\n\n"
        "These results use the validation set employed for model selection.\n"
        "They are not independent final test results.\n"
        "Pneumonia scores are not calibrated clinical probabilities.\n"
        "The test set was not used.\n"
    )

    (out / "evaluation_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"Saved evaluation results to: {out}")


if __name__ == "__main__":
    main()