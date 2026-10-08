"""
train_resnet50.py

Train an ImageNet-pretrained ResNet-50 for binary classification.

Stage 1:
    Train only the new classifier for 3 epochs.

Stage 2:
    Fine-tune the final ResNet block and classifier.

Run:
    .\\.venv\\Scripts\\python.exe .\\train_resnet50.py

Outputs:
    results/resnet50/<run timestamp>/
        best_model.pt
        training_history.csv
        learning_curves.png
        config.json
        training_summary.txt
        train.csv
        val.csv

The initial model is unchanged.
The test set is not used.
"""

import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Reuse the original image loader, which converts images to grayscale.
from train_initial import XRayDataset


ROOT = Path(__file__).resolve().parent

# Use the exact assignments saved with your initial experiment.
INITIAL_RUN = (
    ROOT / "results" / "initial" / "20261006_222400_613495"
)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8

MAX_EPOCHS = 30
HEAD_EPOCHS = 3
PATIENCE = 5

HEAD_LEARNING_RATE = 0.001
BLOCK_LEARNING_RATE = 0.0001

# Normalization associated with the ImageNet-pretrained weights.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class ResNet50CNN(nn.Module):
    """Pretrained ResNet-50 with a binary classification head."""

    def __init__(self, pretrained=True):
        super().__init__()

        # Download weights when constructing a new training model.
        # Evaluation scripts can use pretrained=False and load saved weights.
        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )
        self.network = resnet50(weights=weights)

        # Freeze all original parameters initially.
        for parameter in self.network.parameters():
            parameter.requires_grad = False

        # Replace the ImageNet classifier with one pneumonia logit.
        features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(features, 1),
        )

    def enable_final_block(self):
        """Allow updates to the final ResNet block."""
        for parameter in self.network.layer4.parameters():
            parameter.requires_grad = True

    def set_training_stage(self, stage):
        """Keep frozen layers and their batch statistics fixed."""

        # Start with all modules in evaluation mode.
        self.eval()

        # Enable classifier dropout during training.
        self.network.fc.train()

        # Update final-block batch statistics only during fine-tuning.
        if stage == "finetune":
            self.network.layer4.train()

    def forward(self, images):
        return self.network(images).squeeze(1)


def run_epoch(
    model,
    loader,
    device,
    class_weights,
    stage,
    optimizer=None,
    scaler=None,
):
    """Run training or validation and return classification metrics."""

    training = optimizer is not None

    if training:
        model.set_training_stage(stage)
    else:
        model.eval()

    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    total_loss = 0.0
    targets = []
    scores = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            # Mixed precision reduces GPU memory use during training.
            # Validation uses float32 for consistent later evaluation.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=training,
            ):
                logits = model(images)
                individual_losses = loss_function(logits, labels)
                sample_weights = class_weights[labels.long()]
                loss = (individual_losses * sample_weights).mean()

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "A non-finite loss occurred. Stop and inspect the run."
                )

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(labels)

            targets.extend(labels.detach().cpu().tolist())
            scores.extend(
                torch.sigmoid(logits.detach().float()).cpu().tolist()
            )

    targets = np.asarray(targets, dtype=int)
    scores = np.asarray(scores)
    predictions = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "loss": total_loss / len(loader.dataset),
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
        "auc": float(roc_auc_score(targets, scores)),
    }


def main():
    # Set random seeds and deterministic convolution settings.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Use your GPU-enabled .venv."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # Load the training and validation tables from the initial run.
    train_path = INITIAL_RUN / "train.csv"
    val_path = INITIAL_RUN / "val.csv"

    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit(
            f"Split files not found in: {INITIAL_RUN}\n"
            "Check INITIAL_RUN at the top of this script."
        )

    train_table = pd.read_csv(train_path)
    val_table = pd.read_csv(val_path)

    for table in [train_table, val_table]:
        if table.empty or set(table["target"]) != {0, 1}:
            raise SystemExit("Each split must contain both classes.")

        if table["path"].duplicated().any():
            raise SystemExit("A split contains repeated paths.")

        for path in table["path"]:
            if not (ROOT / path).is_file():
                raise SystemExit(f"Image not found: {ROOT / path}")

    # Verify separation under the existing grouping method.
    for column in ["path", "md5", "group_id"]:
        overlap = (
            set(train_table[column]) & set(val_table[column])
        )

        if overlap:
            raise SystemExit(f"Split overlap found in {column}.")

    # Create a separate folder for this transfer-learning experiment.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "results" / "resnet50" / run_name
    out.mkdir(parents=True, exist_ok=False)

    train_table.to_csv(out / "train.csv", index=False)
    val_table.to_csv(out / "val.csv", index=False)

    # Repeat grayscale across three channels for ResNet-50.
    # Retain the same square-resize geometry and training augmentation.
    train_transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0,
        ),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    # Validation has no random augmentation.
    val_transform = v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    train_dataset = XRayDataset(train_table, train_transform)
    val_dataset = XRayDataset(val_table, val_transform)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Derive balanced class weights only from training labels.
    counts = train_table["target"].value_counts()
    weight_values = [
        len(train_table) / (2 * counts[0]),
        len(train_table) / (2 * counts[1]),
    ]
    class_weights = torch.tensor(
        weight_values,
        dtype=torch.float32,
        device=device,
    )

    # Record settings needed to reproduce the experiment.
    config = {
        "architecture": "ResNet50CNN",
        "pretrained_weights": "ResNet50_Weights.IMAGENET1K_V2",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "geometry": "square resize without center cropping",
        "input": "grayscale repeated across three channels",
        "normalization_mean": IMAGE_MEAN,
        "normalization_std": IMAGE_STD,
        "augmentation": {
            "rotation_degrees": 5,
            "translation_fraction": 0.03,
            "training_only": True,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "head_epochs": HEAD_EPOCHS,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "final_block_learning_rate": BLOCK_LEARNING_RATE,
        "patience": PATIENCE,
        "optimizer": "Adam",
        "optimizer_reset_at_finetuning": True,
        "fine_tuned_modules": ["network.layer4", "network.fc"],
        "frozen_batchnorm_statistics": "fixed outside layer4",
        "classifier_dropout": 0.3,
        "class_weights": [float(value) for value in weight_values],
        "class_mapping": {"NORMAL": 0, "PNEUMONIA": 1},
        "decision_threshold": 0.5,
        "checkpoint_selection": "lowest weighted validation loss",
        "mixed_precision_training": True,
        "validation_precision": "float32",
        "train_images": len(train_table),
        "val_images": len(val_table),
        "initial_run": INITIAL_RUN.name,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "split_file_sha256": {
            "train.csv": hashlib.sha256(
                train_path.read_bytes()
            ).hexdigest(),
            "val.csv": hashlib.sha256(
                val_path.read_bytes()
            ).hexdigest(),
        },
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(
        "Loading ImageNet-pretrained ResNet-50 weights...",
        flush=True,
    )
    model = ResNet50CNN(pretrained=True).to(device)

    # Stage 1 updates only the new classification head.
    optimizer = torch.optim.Adam(
        model.network.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
    )
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_loss = float("inf")
    best_epoch = None
    best_stage = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    print(
        f"Training: {len(train_table)} images | "
        f"Validation: {len(val_table)} images",
        flush=True,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        stage = "head" if epoch <= HEAD_EPOCHS else "finetune"

        if epoch == HEAD_EPOCHS + 1:
            # Stage 2 updates the final ResNet block and classifier.
            model.enable_final_block()

            optimizer = torch.optim.Adam([
                {
                    "params": model.network.layer4.parameters(),
                    "lr": BLOCK_LEARNING_RATE,
                },
                {
                    "params": model.network.fc.parameters(),
                    "lr": HEAD_LEARNING_RATE,
                },
            ])

            # Give fine-tuning its own patience window.
            # Retain the overall best checkpoint from either stage.
            epochs_without_improvement = 0
            print("Starting final-block fine-tuning.", flush=True)

        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            stage,
            optimizer,
            scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            class_weights,
            stage,
        )

        # Record and save metrics after every epoch.
        row = {
            "epoch": epoch,
            "stage": stage,
            "seconds": time.perf_counter() - epoch_start,
        }

        for name, value in train_metrics.items():
            row[f"train_{name}"] = value

        for name, value in val_metrics.items():
            row[f"val_{name}"] = value

        history.append(row)
        pd.DataFrame(history).to_csv(
            out / "training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} [{stage}] | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.3f} | "
            f"Val recall: {val_metrics['recall']:.3f} | "
            f"Val specificity: {val_metrics['specificity']:.3f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Time: {row['seconds']:.1f}s",
            flush=True,
        )

        # Save the model whenever weighted validation loss improves.
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_stage = stage
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "stage": stage,
                    "validation_metrics": val_metrics,
                    "config": config,
                },
                out / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        # Apply early stopping only during fine-tuning.
        if stage == "finetune":
            if epochs_without_improvement >= PATIENCE:
                print(
                    "Early stopping: validation loss stopped improving."
                )
                break

    # Plot learning curves and mark the fine-tuning transition.
    history_table = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for column, metric in enumerate(["loss", "accuracy"]):
        axes[column].plot(
            history_table["epoch"],
            history_table[f"train_{metric}"],
            label="Training",
        )
        axes[column].plot(
            history_table["epoch"],
            history_table[f"val_{metric}"],
            label="Validation",
        )
        axes[column].axvline(
            HEAD_EPOCHS + 0.5,
            linestyle="--",
            color="gray",
            label="Fine-tuning starts",
        )
        axes[column].set_xlabel("Epoch")
        axes[column].set_ylabel(
            "Weighted loss" if metric == "loss" else "Accuracy"
        )
        axes[column].legend()

    plt.tight_layout()
    plt.savefig(out / "learning_curves.png", dpi=200)
    plt.close(fig)

    # Save and print the final training summary.
    summary = (
        f"Best epoch: {best_epoch}\n"
        f"Best stage: {best_stage}\n"
        f"Best validation loss: {best_loss:.4f}\n"
        f"Epochs completed: {len(history)}\n"
        f"Total training time: "
        f"{(time.perf_counter() - training_start) / 60:.1f} minutes\n"
        "Model selection used validation loss.\n"
        "The test set was not used.\n"
    )

    (out / "training_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + summary)
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()