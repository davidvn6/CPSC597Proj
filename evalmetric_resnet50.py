"""
evalmetric_resnet50.py

Evaluate the saved ResNet-50 on validation images.

Run:
    .\\.venv\\Scripts\\python.exe .\\evalmetric_resnet50.py

Outputs inside the selected ResNet-50 run:
    validation_evaluation/
        predictions.csv
        metrics.json
        evaluation_summary.txt
        false_negatives.csv
        false_positives.csv
        confusion_matrix.png
        roc_curve.png
        false_negative_examples.png
        false_positive_examples.png
        subtype_results.csv
        image_size_results.csv
        image_size_groups.json
        model_comparison.csv
        model_comparison_summary.txt

The model_comparison files require the initial model's predictions.
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
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Reuse the trained architecture and original image loader.
from train_resnet50 import ResNet50CNN
from train_initial import XRayDataset

# Reuse the function that displays false-positive/negative examples.
from evalmetric_initial import save_error_examples


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "resnet50"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def add_outcomes(table):
    """Assign TP, TN, FP, or FN to every prediction."""

    outcomes = []

    for _, row in table.iterrows():
        actual = int(row["target"])
        predicted = int(row["predicted_target"])

        if actual == 1 and predicted == 1:
            outcomes.append("TP")
        elif actual == 0 and predicted == 0:
            outcomes.append("TN")
        elif actual == 0 and predicted == 1:
            outcomes.append("FP")
        else:
            outcomes.append("FN")

    table["outcome"] = outcomes
    return table


def summarize_sizes(table, model_name):
    """Summarize performance in the predefined image-size groups."""

    rows = []

    for group_name, group in table.groupby("size_group"):
        pneumonia = group[group["target"] == 1]
        normal = group[group["target"] == 0]

        fn = int((pneumonia["predicted_target"] == 0).sum())
        fp = int((normal["predicted_target"] == 1).sum())
        errors = int((~group["correct"]).sum())

        rows.append({
            "model": model_name,
            "size_group": group_name,
            "images": len(group),
            "normal_images": len(normal),
            "pneumonia_images": len(pneumonia),
            "errors": errors,
            "error_rate": errors / len(group),
            "false_negatives": fn,
            "false_positives": fp,
            "pneumonia_recall": (
                1 - fn / len(pneumonia)
                if len(pneumonia)
                else None
            ),
            "specificity": (
                1 - fp / len(normal)
                if len(normal)
                else None
            ),
        })

    return pd.DataFrame(rows)


def main():
    # Select the latest timestamped ResNet-50 run with a checkpoint.
    checkpoints = sorted(RUNS.glob("*/best_model.pt"))

    if not checkpoints:
        raise SystemExit(f"No checkpoint found inside: {RUNS}")

    checkpoint_path = checkpoints[-1]
    run_folder = checkpoint_path.parent

    print(f"Selected training run: {run_folder}", flush=True)

    # Construct the architecture without downloading pretrained weights again.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    if config["architecture"] != "ResNet50CNN":
        raise SystemExit("This checkpoint is not a ResNet50CNN.")

    # Verify and load the validation CSV saved with this training run.
    val_path = run_folder / "val.csv"
    actual_hash = hashlib.sha256(val_path.read_bytes()).hexdigest()

    if actual_hash != config["split_file_sha256"]["val.csv"]:
        raise SystemExit("The validation CSV has changed since training.")

    table = pd.read_csv(val_path).reset_index(drop=True)

    if table.empty or set(table["target"]) != {0, 1}:
        raise SystemExit("Validation must contain both classes.")

    if table["path"].duplicated().any():
        raise SystemExit("Validation contains repeated paths.")

    # Reproduce ResNet-50 validation preprocessing exactly.
    size = config["image_size"]
    transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=config["normalization_mean"],
            std=config["normalization_std"],
        ),
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

    model = ResNet50CNN(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Evaluating epoch {checkpoint['epoch']} "
        f"on {len(table)} validation images using {device}.",
        flush=True,
    )

    # Use float32 inference, matching training-time validation.
    scores = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            scores.extend(torch.sigmoid(logits).cpu().tolist())

    scores = np.asarray(scores)
    targets = table["target"].to_numpy(dtype=int)
    threshold = config["decision_threshold"]
    predictions = (scores >= threshold).astype(int)

    table["pneumonia_score"] = scores
    table["predicted_target"] = predictions
    table["predicted_label"] = table["predicted_target"].map({
        0: "NORMAL",
        1: "PNEUMONIA",
    })
    table["correct"] = predictions == targets
    table = add_outcomes(table)

    # Calculate overall validation metrics.
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
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

    # Define size groups using original image pixel area.
    # Both models will use these same validation-derived boundaries.
    area = table["width"].astype(float) * table["height"].astype(float)

    size_codes, size_edges = pd.qcut(
        area,
        q=4,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    table["size_group"] = [
        f"Q{int(code) + 1}" for code in size_codes
    ]

    (out / "image_size_groups.json").write_text(
        json.dumps({
            "measure": "original width multiplied by height",
            "boundaries": [float(value) for value in size_edges],
            "definition": "validation pixel-area quantiles",
            "note": "Image size is not a clinical severity measure.",
        }, indent=2),
        encoding="utf-8",
    )

    # Record all predictions and the error inventories.
    table.to_csv(out / "predictions.csv", index=False)

    false_negatives = table[table["outcome"] == "FN"].sort_values(
        "pneumonia_score"
    )
    false_positives = table[table["outcome"] == "FP"].sort_values(
        "pneumonia_score",
        ascending=False,
    )

    false_negatives.to_csv(out / "false_negatives.csv", index=False)
    false_positives.to_csv(out / "false_positives.csv", index=False)

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Summarize pneumonia recall by filename-derived subtype.
    subtype_rows = []

    pneumonia_table = table[table["target"] == 1].copy()
    pneumonia_table["ptype"] = pneumonia_table["ptype"].fillna("unknown")

    for subtype, group in pneumonia_table.groupby("ptype"):
        missed = int((group["predicted_target"] == 0).sum())

        subtype_rows.append({
            "model": "ResNet50",
            "subtype": subtype,
            "images": len(group),
            "false_negatives": missed,
            "recall": 1 - missed / len(group),
        })

    size_results = summarize_sizes(table, "ResNet50")

    # Compare with the initial model on the exact same validation images.
    initial_path = (
        ROOT / "results" / "initial" / config["initial_run"]
        / "validation_evaluation" / "predictions.csv"
    )

    comparison_summary = (
        "Initial-model predictions were not found; comparison skipped."
    )

    if initial_path.is_file():
        initial = pd.read_csv(initial_path)

        if initial["path"].duplicated().any():
            raise SystemExit("Initial predictions contain repeated paths.")

        if set(initial["path"]) != set(table["path"]):
            raise SystemExit("The models use different validation images.")

        # Align initial predictions to ResNet-50's CSV order.
        initial = initial.set_index("path").loc[
            table["path"]
        ].reset_index()

        if not np.array_equal(
            initial["target"].to_numpy(),
            targets,
        ):
            raise SystemExit("Validation labels differ between models.")

        # Use the same image-size groups for both models.
        initial["size_group"] = table["size_group"].to_numpy()
        initial["correct"] = (
            initial["predicted_target"].to_numpy() == targets
        )

        size_results = pd.concat([
            summarize_sizes(initial, "InitialCNN"),
            size_results,
        ], ignore_index=True)

        for subtype, group in initial[
            initial["target"] == 1
        ].groupby("ptype", dropna=False):
            missed = int((group["predicted_target"] == 0).sum())

            subtype_rows.append({
                "model": "InitialCNN",
                "subtype": (
                    subtype if pd.notna(subtype) else "unknown"
                ),
                "images": len(group),
                "false_negatives": missed,
                "recall": 1 - missed / len(group),
            })

        initial_correct = initial["correct"].to_numpy(dtype=bool)
        resnet_correct = table["correct"].to_numpy(dtype=bool)

        comparison = table[[
            "path", "label", "target", "ptype", "size_group"
        ]].copy()
        comparison["initial_score"] = initial[
            "pneumonia_score"
        ].to_numpy()
        comparison["resnet50_score"] = scores
        comparison["initial_prediction"] = initial[
            "predicted_target"
        ].to_numpy()
        comparison["resnet50_prediction"] = predictions

        categories = []

        for index in range(len(table)):
            if initial_correct[index] and resnet_correct[index]:
                categories.append("both_correct")
            elif not initial_correct[index] and not resnet_correct[index]:
                categories.append("both_wrong")
            elif resnet_correct[index]:
                categories.append("corrected_by_resnet50")
            else:
                categories.append("new_error_in_resnet50")

        comparison["comparison_category"] = categories
        comparison.to_csv(out / "model_comparison.csv", index=False)

        comparison_summary = (
            "Paired validation comparison:\n"
            + comparison["comparison_category"].value_counts().to_string()
        )

    (out / "model_comparison_summary.txt").write_text(
        comparison_summary,
        encoding="utf-8",
    )

    subtype_results = pd.DataFrame(subtype_rows)
    subtype_results.to_csv(out / "subtype_results.csv", index=False)
    size_results.to_csv(out / "image_size_results.csv", index=False)

    # Plot the confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(
        f"ResNet-50: validation confusion matrix\n"
        f"Epoch {checkpoint['epoch']}, threshold {threshold}"
    )
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Plot the ROC curve.
    fpr, tpr, thresholds = roc_curve(targets, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ResNet-50: validation ROC curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=200)
    plt.close(fig)

    # Show up to six examples from each error category.
    save_error_examples(
        false_negatives,
        "ResNet-50 validation false negatives",
        out / "false_negative_examples.png",
    )
    save_error_examples(
        false_positives,
        "ResNet-50 validation false positives",
        out / "false_positive_examples.png",
    )

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
        f"Subtype results:\n{subtype_results.round(4).to_string(index=False)}"
        f"\n\n{comparison_summary}\n\n"
        "Subtype labels come from filenames.\n"
        "Image-size analyses are exploratory.\n"
        "Scores are not calibrated clinical probabilities.\n"
        "Validation was used for model selection.\n"
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