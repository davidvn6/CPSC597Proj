"""
evalmetric_resnet50_letterbox.py

Evaluate the saved letterbox-trained ResNet-50 on validation.

Uses matching preprocessing:
    grayscale -> three channels -> aspect-preserving resize ->
    centered black padding -> ImageNet normalization

Outputs:
    results/resnet50_letterbox/<run>/validation_evaluation/
        predictions.csv
        metrics.json
        classification_report.txt
        false_negatives.csv
        false_positives.csv
        confusion_matrix.png
        roc_curve.png
        subtype_results.csv
        square_model_comparison.csv
        model_metrics_comparison.csv
        summary.txt

Run:
    .\\.venv\\Scripts\\python.exe .\\evalmetric_resnet50_letterbox.py

The test set is not used.
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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from train_initial import XRayDataset
from train_resnet50 import ResNet50CNN
from resize_test_resnet50 import AspectPreservingResize
from gradcam_resnet50 import get_validation_hash, sha256_file


# ------------------------------------------------------------
# Select the exact runs being compared
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

LETTERBOX_RUN = (
    ROOT / "results" / "resnet50_letterbox"
    / "20261007_213723_403333"
)

SQUARE_RUN = (
    ROOT / "results" / "resnet50"
    / "20261007_172849_403643"
)

INITIAL_RUN = (
    ROOT / "results" / "initial"
    / "20261006_222400_613495"
)

SEED = 42


def metrics_from_scores(targets, scores, threshold):
    """Calculate binary classification metrics."""
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predicted,
        labels=[0, 1],
    ).ravel()

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
    }


def load_reference_predictions(run, validation):
    """Load another model's predictions and align them by image path."""
    prediction_path = run / "validation_evaluation" / "predictions.csv"

    if not prediction_path.exists():
        raise SystemExit(
            f"Reference predictions not found:\n{prediction_path}"
        )

    table = pd.read_csv(prediction_path)

    if not table["path"].is_unique:
        raise SystemExit(f"Duplicate prediction paths in:\n{prediction_path}")

    if set(table["path"]) != set(validation["path"]):
        raise SystemExit(
            f"Reference model uses different validation images:\n{run}"
        )

    table = (
        table.set_index("path")
        .loc[validation["path"]]
        .reset_index()
    )

    if not np.array_equal(
        table["target"].to_numpy(dtype=int),
        validation["target"].to_numpy(dtype=int),
    ):
        raise SystemExit(f"Reference validation labels differ:\n{run}")

    return table


def main():
    # --------------------------------------------------------
    # Reproducibility settings
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------------------------------------
    # Load and verify the letterbox checkpoint
    # --------------------------------------------------------

    checkpoint_path = LETTERBOX_RUN / "best_model.pt"
    validation_path = LETTERBOX_RUN / "val.csv"

    for path in [checkpoint_path, validation_path]:
        if not path.exists():
            raise SystemExit(f"Required file not found:\n{path}")

    print(f"Selected training run: {LETTERBOX_RUN}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    if config.get("architecture") != "ResNet50CNN":
        raise SystemExit("The checkpoint is not a ResNet50CNN model.")

    if config.get("preprocessing") != "aspect_preserving_black_letterbox":
        raise SystemExit(
            "The checkpoint does not identify black-letterbox preprocessing."
        )

    if sha256_file(validation_path) != get_validation_hash(config):
        raise SystemExit(
            "Validation assignments do not match the checkpoint configuration."
        )

    validation = pd.read_csv(validation_path)

    if not validation["path"].is_unique:
        raise SystemExit("Duplicate paths in validation assignments.")

    targets = validation["target"].to_numpy(dtype=int)

    if set(targets) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    # Verify comparison inputs before running inference.
    square_predictions = load_reference_predictions(
        SQUARE_RUN,
        validation,
    )
    initial_predictions = load_reference_predictions(
        INITIAL_RUN,
        validation,
    )

    # --------------------------------------------------------
    # Recreate the letterbox validation preprocessing
    # --------------------------------------------------------

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    threshold = float(config["decision_threshold"])

    transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        AspectPreservingResize(image_size, "black"),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=config["normalization_mean"],
            std=config["normalization_std"],
        ),
    ])

    loader = DataLoader(
        XRayDataset(validation, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet50CNN(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use the same training-derived class weights for validation loss.
    class_weights = torch.tensor(
        config["class_weights"],
        dtype=torch.float32,
        device=device,
    )
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

    print(
        f"Evaluating epoch {checkpoint['epoch']} on "
        f"{len(validation)} validation images using {device}."
    )

    # --------------------------------------------------------
    # Run full-precision validation inference
    # --------------------------------------------------------

    score_batches = []
    total_loss = 0.0
    total_images = 0

    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(device, dtype=torch.float32)
            batch_targets = batch_targets.to(
                device,
                dtype=torch.float32,
            ).reshape(-1)

            logits = model(images).reshape(-1)

            per_image_loss = loss_function(logits, batch_targets)
            weights = torch.where(
                batch_targets == 1,
                class_weights[1],
                class_weights[0],
            )
            loss = (per_image_loss * weights).mean()

            number = len(batch_targets)
            total_loss += float(loss.item()) * number
            total_images += number

            score_batches.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    scores = np.concatenate(score_batches)

    if len(scores) != len(validation) or not np.isfinite(scores).all():
        raise SystemExit("Invalid validation prediction scores.")

    predicted = (scores >= threshold).astype(int)

    metrics = metrics_from_scores(targets, scores, threshold)
    metrics.update({
        "weighted_validation_loss": total_loss / total_images,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_images": len(validation),
        "decision_threshold": threshold,
    })

    # --------------------------------------------------------
    # Save predictions and error lists
    # --------------------------------------------------------

    output = LETTERBOX_RUN / "validation_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    predictions = validation.copy()
    predictions["pneumonia_score"] = scores
    predictions["predicted_target"] = predicted
    predictions["predicted_label"] = np.where(
        predicted == 1,
        "PNEUMONIA",
        "NORMAL",
    )
    predictions["correct"] = predicted == targets

    predictions["outcome"] = np.select(
        [
            (targets == 1) & (predicted == 1),
            (targets == 0) & (predicted == 0),
            (targets == 0) & (predicted == 1),
            (targets == 1) & (predicted == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )

    predictions.to_csv(output / "predictions.csv", index=False)

    predictions[predictions["outcome"] == "FN"].sort_values(
        "pneumonia_score"
    ).to_csv(output / "false_negatives.csv", index=False)

    predictions[predictions["outcome"] == "FP"].sort_values(
        "pneumonia_score",
        ascending=False,
    ).to_csv(output / "false_positives.csv", index=False)

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    report = classification_report(
        targets,
        predicted,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4,
        zero_division=0,
    )

    (output / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save confusion matrix and ROC curve
    # --------------------------------------------------------

    matrix = confusion_matrix(targets, predicted, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=["NORMAL", "PNEUMONIA"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Letterbox ResNet-50: validation")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        targets,
        scores,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Letterbox ResNet-50: validation ROC")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------
    # Compare all three models at the fixed threshold of 0.5
    # --------------------------------------------------------

    model_scores = {
        "InitialCNN": initial_predictions[
            "pneumonia_score"
        ].to_numpy(dtype=float),
        "ResNet50_square": square_predictions[
            "pneumonia_score"
        ].to_numpy(dtype=float),
        "ResNet50_letterbox": scores,
    }

    comparison_rows = []
    subtype_rows = []

    # Use a common threshold for this preprocessing comparison.
    comparison_threshold = 0.5

    # Subtype labels are filename-derived metadata, not new diagnoses.
    subtypes = validation["ptype"].fillna("unknown").to_numpy()

    for model_name, model_score in model_scores.items():
        model_prediction = (
            model_score >= comparison_threshold
        ).astype(int)

        comparison_rows.append({
            "model": model_name,
            "threshold": comparison_threshold,
            **metrics_from_scores(
                targets,
                model_score,
                comparison_threshold,
            ),
        })

        for subtype in ["bacterial", "viral"]:
            selected = (targets == 1) & (subtypes == subtype)
            number = int(selected.sum())

            if number == 0:
                continue

            false_negatives = int(
                np.sum(model_prediction[selected] == 0)
            )

            subtype_rows.append({
                "model": model_name,
                "subtype": subtype,
                "images": number,
                "false_negatives": false_negatives,
                "recall": 1 - false_negatives / number,
            })

    model_comparison = pd.DataFrame(comparison_rows)
    subtype_results = pd.DataFrame(subtype_rows)

    model_comparison.to_csv(
        output / "model_metrics_comparison.csv",
        index=False,
    )
    subtype_results.to_csv(
        output / "subtype_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compare individual errors between the two ResNet models
    # --------------------------------------------------------

    square_scores = model_scores["ResNet50_square"]
    square_predicted = (
        square_scores >= comparison_threshold
    ).astype(int)
    letterbox_predicted = (
        scores >= comparison_threshold
    ).astype(int)

    square_correct = square_predicted == targets
    letterbox_correct = letterbox_predicted == targets

    paired = validation.copy()
    paired["square_score"] = square_scores
    paired["letterbox_score"] = scores
    paired["square_prediction"] = square_predicted
    paired["letterbox_prediction"] = letterbox_predicted

    paired["comparison_category"] = np.select(
        [
            square_correct & letterbox_correct,
            ~square_correct & letterbox_correct,
            square_correct & ~letterbox_correct,
            ~square_correct & ~letterbox_correct,
        ],
        [
            "both_correct",
            "corrected_by_letterbox",
            "new_error_in_letterbox",
            "both_wrong",
        ],
        default="UNKNOWN",
    )

    paired.to_csv(
        output / "square_model_comparison.csv",
        index=False,
    )

    paired_counts = paired["comparison_category"].value_counts()

    # --------------------------------------------------------
    # Print and save the summary
    # --------------------------------------------------------

    summary = "\n".join([
        f"Checkpoint epoch: {checkpoint['epoch']}",
        f"Validation images: {len(validation)}",
        f"Decision threshold: {threshold}",
        "Preprocessing: aspect-preserving resize with black padding",
        "",
        report,
        f"Weighted validation loss: "
        f"{metrics['weighted_validation_loss']:.4f}",
        f"Specificity: {metrics['specificity']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"True negatives: {metrics['true_negatives']}",
        f"False positives: {metrics['false_positives']}",
        f"False negatives: {metrics['false_negatives']}",
        f"True positives: {metrics['true_positives']}",
        "",
        "Three-model comparison at threshold 0.5:",
        model_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ),
        "",
        "Subtype results:",
        subtype_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ),
        "",
        "Paired square-versus-letterbox comparison:",
        paired_counts.to_string(),
        "",
        "Subtype labels come from filenames.",
        "The comparison includes aspect preservation and padding together.",
        "Each training condition has one run.",
        "Validation was used for checkpoint selection.",
        "These are not independent final test results.",
        "Scores are not calibrated clinical probabilities.",
        "The test set was not used.",
    ])

    (output / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"\nSaved evaluation results to: {output}")


if __name__ == "__main__":
    main()