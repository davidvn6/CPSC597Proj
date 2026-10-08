"""
train_resnet50_letterbox.py

Train an ImageNet-pretrained ResNet-50 using aspect-preserving resizing
with centered black padding.

Training:
    - Same train/validation assignments as the square-resize model.
    - Three epochs training the classification head.
    - Then fine-tune the final residual block and classification head.
    - Select the checkpoint using lowest weighted validation loss.
    - Stop after five fine-tuning epochs without improvement.

Run:
    .\\.venv\\Scripts\\python.exe .\\train_resnet50_letterbox.py

Outputs:
    results/resnet50_letterbox/<timestamp>/

The test set is not used.
"""

import copy
import json
import random
import shutil
import time
from datetime import datetime
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
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2

from train_initial import XRayDataset
from train_resnet50 import ResNet50CNN
from resize_test_resnet50 import AspectPreservingResize
from gradcam_resnet50 import get_validation_hash, sha256_file


# ------------------------------------------------------------
# Paths and training settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# Use the same saved assignments as the completed square-resize run.
SOURCE_RUN = (
    ROOT / "results" / "resnet50" / "20261007_172849_403643"
)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8

MAX_EPOCHS = 30
HEAD_EPOCHS = 3
PATIENCE = 5

HEAD_LEARNING_RATE = 0.001
BLOCK_LEARNING_RATE = 0.0001
FINETUNE_HEAD_LEARNING_RATE = 0.001

THRESHOLD = 0.5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed():
    """Set random seeds and deterministic cuDNN options."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_training_mode(model, stage):
    """
    Keep frozen layers and their batch-normalization statistics fixed.

    Only the classification head is in training mode during the head stage.
    The final residual block also enters training mode during fine-tuning.
    """
    model.eval()
    model.network.fc.train()

    if stage == "finetune":
        model.network.layer4.train()


def weighted_loss(logits, targets, loss_function, class_weights):
    """Apply training-derived class weights to per-image BCE loss."""
    per_image_loss = loss_function(logits, targets)

    weights = torch.where(
        targets == 1,
        class_weights[1],
        class_weights[0],
    )

    return (per_image_loss * weights).mean()


def evaluate(model, loader, device, loss_function, class_weights):
    """Evaluate validation loss and metrics in full precision."""
    model.eval()

    total_loss = 0.0
    total_images = 0
    target_batches = []
    score_batches = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32).reshape(-1)

            logits = model(images).reshape(-1)

            loss = weighted_loss(
                logits,
                targets,
                loss_function,
                class_weights,
            )

            number = len(targets)
            total_loss += float(loss.item()) * number
            total_images += number

            target_batches.append(targets.cpu().numpy())
            score_batches.append(torch.sigmoid(logits).cpu().numpy())

    targets = np.concatenate(target_batches).astype(int)
    scores = np.concatenate(score_batches)
    predicted = (scores >= THRESHOLD).astype(int)

    if not np.isfinite(scores).all():
        raise SystemExit("Non-finite validation scores encountered.")

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "loss": total_loss / total_images,
        "accuracy": float(accuracy_score(targets, predicted)),
        "recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(targets, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, scores)),
    }


def save_learning_curves(history, output):
    """Save loss and validation-metric curves for the report."""
    table = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        table["epoch"], table["train_loss"], label="Training"
    )
    axes[0].plot(
        table["epoch"], table["val_loss"], label="Validation"
    )
    axes[0].set_title("Weighted loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    for column, label in [
        ("val_accuracy", "Accuracy"),
        ("val_recall", "Recall"),
        ("val_specificity", "Specificity"),
        ("val_roc_auc", "ROC-AUC"),
    ]:
        axes[1].plot(table["epoch"], table[column], label=label)

    axes[1].set_title("Validation metrics at threshold 0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    for axis in axes:
        axis.axvline(
            HEAD_EPOCHS + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=200)
    plt.close(fig)


def main():
    set_seed()

    # --------------------------------------------------------
    # Verify and load the original train/validation assignments
    # --------------------------------------------------------

    source_checkpoint_path = SOURCE_RUN / "best_model.pt"
    train_path = SOURCE_RUN / "train.csv"
    validation_path = SOURCE_RUN / "val.csv"

    for path in [source_checkpoint_path, train_path, validation_path]:
        if not path.exists():
            raise SystemExit(f"Required source file not found:\n{path}")

    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    source_config = source_checkpoint["config"]

    if sha256_file(validation_path) != get_validation_hash(source_config):
        raise SystemExit("Source validation assignments have changed.")

    # The new run uses exact copies of these saved assignments.
    train_table = pd.read_csv(train_path)
    validation_table = pd.read_csv(validation_path)

    for name, table in [
        ("Training", train_table),
        ("Validation", validation_table),
    ]:
        if not table["path"].is_unique:
            raise SystemExit(f"{name} contains duplicate paths.")

        if set(table["target"].astype(int)) != {0, 1}:
            raise SystemExit(f"{name} must contain both classes.")

    if set(train_table["path"]) & set(validation_table["path"]):
        raise SystemExit("Training and validation image paths overlap.")

    for table in [train_table, validation_table]:
        for relative_path in table["path"]:
            if not (ROOT / relative_path).exists():
                raise SystemExit(f"Image not found:\n{ROOT / relative_path}")

    # --------------------------------------------------------
    # Create transforms
    # --------------------------------------------------------

    # Keep the same augmentation parameters as the square model.
    # The letterbox operation replaces the square resize operation.
    train_transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        AspectPreservingResize(IMAGE_SIZE, "black"),
        v2.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])

    validation_transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        AspectPreservingResize(IMAGE_SIZE, "black"),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])

    train_loader = DataLoader(
        XRayDataset(train_table, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        XRayDataset(validation_table, validation_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Initialize a fresh ImageNet-pretrained model
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    print("Loading ImageNet-pretrained ResNet-50 weights...")

    # Start from ImageNet weights, not the square-trained checkpoint.
    model = ResNet50CNN(pretrained=True).to(device)

    # Explicitly freeze the backbone and enable the classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.network.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        model.network.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
    )

    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Calculate class weights from TRAINING labels only.
    normal_count = int((train_table["target"] == 0).sum())
    pneumonia_count = int((train_table["target"] == 1).sum())
    training_count = len(train_table)

    class_weights = torch.tensor(
        [
            training_count / (2 * normal_count),
            training_count / (2 * pneumonia_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    # Mixed precision is used for CUDA training only.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --------------------------------------------------------
    # Create a separate run folder and save configuration
    # --------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "results" / "resnet50_letterbox" / timestamp
    output.mkdir(parents=True, exist_ok=True)

    shutil.copy2(train_path, output / "train.csv")
    shutil.copy2(validation_path, output / "val.csv")

    # Retain compatible metadata, then record this experiment's settings.
    config = copy.deepcopy(source_config)
    config.update({
        "architecture": "ResNet50CNN",
        "preprocessing": "aspect_preserving_black_letterbox",
        "comparison_run": str(SOURCE_RUN.relative_to(ROOT)),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "head_epochs": HEAD_EPOCHS,
        "patience": PATIENCE,
        "decision_threshold": THRESHOLD,
        "normalization_mean": MEAN,
        "normalization_std": STD,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "block_learning_rate": BLOCK_LEARNING_RATE,
        "finetune_head_learning_rate": FINETUNE_HEAD_LEARNING_RATE,
        "class_weights": class_weights.cpu().tolist(),
        "split_file_sha256": {
            "train": sha256_file(output / "train.csv"),
            "val": sha256_file(output / "val.csv"),
        },
    })

    (output / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(
        f"Training: {len(train_table)} images | "
        f"Validation: {len(validation_table)} images"
    )
    print("Preprocessing: aspect-preserving resize with black padding")

    # --------------------------------------------------------
    # Train the head, then fine-tune the final residual block
    # --------------------------------------------------------

    history = []
    best_loss = float("inf")
    best_epoch = None
    best_stage = None
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()

        stage = "head" if epoch <= HEAD_EPOCHS else "finetune"

        if epoch == HEAD_EPOCHS + 1:
            print("Starting final-block fine-tuning.")

            for parameter in model.network.layer4.parameters():
                parameter.requires_grad = True

            # Reset Adam when entering the fine-tuning stage.
            optimizer = torch.optim.Adam([
                {
                    "params": model.network.layer4.parameters(),
                    "lr": BLOCK_LEARNING_RATE,
                },
                {
                    "params": model.network.fc.parameters(),
                    "lr": FINETUNE_HEAD_LEARNING_RATE,
                },
            ])

            epochs_without_improvement = 0

        set_training_mode(model, stage)

        total_loss = 0.0
        total_images = 0

        for images, targets in train_loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32).reshape(-1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(images).reshape(-1)

                loss = weighted_loss(
                    logits,
                    targets,
                    loss_function,
                    class_weights,
                )

            if not torch.isfinite(loss):
                raise SystemExit("Non-finite training loss encountered.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            number = len(targets)
            total_loss += float(loss.item()) * number
            total_images += number

        train_loss = total_loss / total_images

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            loss_function,
            class_weights,
        )

        if not np.isfinite(validation_metrics["loss"]):
            raise SystemExit("Non-finite validation loss encountered.")

        epoch_seconds = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_recall": validation_metrics["recall"],
            "val_specificity": validation_metrics["specificity"],
            "val_f1": validation_metrics["f1"],
            "val_roc_auc": validation_metrics["roc_auc"],
            "epoch_seconds": epoch_seconds,
        })

        # Save progress after every epoch.
        pd.DataFrame(history).to_csv(
            output / "training_history.csv",
            index=False,
        )

        # Select the best checkpoint across both stages.
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_stage = stage
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "validation_metrics": validation_metrics,
                "config": config,
            }, output / "best_model.pt")

        elif stage == "finetune":
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} [{stage}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {validation_metrics['loss']:.4f} | "
            f"Val accuracy: {validation_metrics['accuracy']:.3f} | "
            f"Val recall: {validation_metrics['recall']:.3f} | "
            f"Val specificity: {validation_metrics['specificity']:.3f} | "
            f"Val AUC: {validation_metrics['roc_auc']:.4f} | "
            f"Time: {epoch_seconds:.1f}s"
        )

        if (
            stage == "finetune"
            and epochs_without_improvement >= PATIENCE
        ):
            print("Early stopping: validation loss stopped improving.")
            break

    # --------------------------------------------------------
    # Save learning curves and the training summary
    # --------------------------------------------------------

    elapsed_minutes = (
        time.perf_counter() - training_start
    ) / 60

    save_learning_curves(history, output)

    summary = "\n".join([
        "ResNet-50 letterbox training",
        "Preprocessing: aspect-preserving resize with black padding",
        f"Training images: {len(train_table)}",
        f"Validation images: {len(validation_table)}",
        f"Best epoch: {best_epoch}",
        f"Best stage: {best_stage}",
        f"Best validation loss: {best_loss:.4f}",
        f"Epochs completed: {len(history)}",
        f"Total training time: {elapsed_minutes:.1f} minutes",
        "",
        "The run started from ImageNet-pretrained weights.",
        "It did not continue training the square-resize checkpoint.",
        "Model selection used weighted validation loss.",
        "Patient independence remains unverified.",
        "This comparison includes aspect preservation and padding together.",
        "The test set was not used.",
    ])

    (output / "training_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + summary)
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()