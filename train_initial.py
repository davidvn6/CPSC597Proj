"""
train_initial.py

Train an initial CNN using the saved training and validation splits.

Run:
    .\\.venv\\Scripts\\python.exe .\\train_initial.py

Outputs:
    results/initial/<run timestamp>/
        best_model.pt
        training_history.csv
        learning_curves.png
        config.json
        train.csv
        val.csv

The test set is not used by this script.
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
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Locate the project folder and split CSVs.
ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "results" / "splits"

# Set the initial experiment settings.
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5


class XRayDataset(Dataset):
    """Load one image and its numeric label from a split table."""

    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = ROOT / row["path"]

        # Apply the recorded orientation and convert to grayscale.
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image = self.transform(image)

        # NORMAL = 0; PNEUMONIA = 1.
        target = torch.tensor(
            float(row["target"]),
            dtype=torch.float32,
        )

        return image, target


class InitialCNN(nn.Module):
    """Four convolution blocks followed by a binary classifier."""

    def __init__(self):
        super().__init__()

        layers = []
        input_channels = 1

        # Increase feature channels as spatial dimensions decrease.
        for output_channels in [16, 32, 64, 128]:
            layers.extend([
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ])
            input_channels = output_channels

        self.features = nn.Sequential(*layers)

        # Average each final feature map into one value.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)

        # Return logits; the loss function handles the sigmoid operation.
        return self.classifier(pooled).squeeze(1)


def run_epoch(model, loader, device, class_weights, optimizer=None):
    """Run one training or validation epoch and calculate metrics."""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    targets = []
    probabilities = []

    # Obtain individual losses so both classes receive balanced weights.
    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    # Enable gradients during training and disable them for validation.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            # Apply weights calculated only from training class counts.
            sample_weights = class_weights[labels.long()]
            losses = loss_function(logits, labels)
            loss = (losses * sample_weights).mean()

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)

            targets.extend(labels.detach().cpu().tolist())
            probabilities.extend(
                torch.sigmoid(logits).detach().cpu().tolist()
            )

    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities)

    # Use a fixed threshold of 0.5 for this initial experiment.
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(
            targets, predictions, zero_division=0
        ),
        "recall": recall_score(
            targets, predictions, zero_division=0
        ),
        "f1": f1_score(targets, predictions, zero_division=0),
        "auc": roc_auc_score(targets, probabilities),
    }


def main():
    # Set random seeds for reproducibility.
    # Exact results may differ across hardware/software versions.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Require GPU availability for this local training setup.
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Run using your GPU-enabled .venv."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # Load only the training and validation assignments.
    train_table = pd.read_csv(SPLITS / "train.csv")
    val_table = pd.read_csv(SPLITS / "val.csv")

    for table in [train_table, val_table]:
        if table.empty or set(table["target"]) != {0, 1}:
            raise SystemExit(
                "Each split must contain both NORMAL and PNEUMONIA."
            )

        if table["path"].duplicated().any():
            raise SystemExit("A split contains repeated image paths.")

    # Confirm training and validation assignments remain disjoint.
    for column in ["path", "md5", "group_id"]:
        overlap = (
            set(train_table[column]) & set(val_table[column])
        )

        if overlap:
            raise SystemExit(
                f"Training/validation overlap found in {column}."
            )

    # Create a separate output folder for each initial experiment.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "results" / "initial" / run_name
    out.mkdir(parents=True, exist_ok=False)

    # Save the exact assignments used in this experiment.
    train_table.to_csv(out / "train.csv", index=False)
    val_table.to_csv(out / "val.csv", index=False)

    # Apply random augmentation only when loading training images.
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True,
        ),
        v2.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Validation uses deterministic resizing and pixel scaling.
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = XRayDataset(train_table, train_transform)
    val_dataset = XRayDataset(val_table, val_transform)

    # Load images in batches instead of holding the dataset in memory.
    # Zero background workers keeps the Windows setup simple.
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

    # Give each class equal total weight in the training loss.
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

    model = InitialCNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Record settings needed to reproduce and describe the experiment.
    config = {
        "architecture": "InitialCNN: channels 16,32,64,128",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "color_mode": "grayscale",
        "pixel_range": [0, 1],
        "resize_method": "resize to square; aspect ratio may change",
        "augmentation": {
            "rotation_degrees": 5,
            "translation_fraction": 0.03,
            "training_only": True,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "optimizer": "Adam",
        "class_weights": [
            float(value) for value in weight_values
        ],
        "class_mapping": {
            "NORMAL": 0,
            "PNEUMONIA": 1,
        },
        "decision_threshold": 0.5,
        "checkpoint_selection": "lowest weighted validation loss",
        "train_images": len(train_table),
        "val_images": len(val_table),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "split_file_sha256": {
            name: hashlib.sha256(
                (SPLITS / name).read_bytes()
            ).hexdigest()
            for name in ["train.csv", "val.csv"]
        },
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    history = []
    best_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    print(
        f"Training: {len(train_table)} images | "
        f"Validation: {len(val_table)} images",
        flush=True,
    )

    # Train and validate once per epoch.
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            class_weights,
        )

        # Record metrics and elapsed time for this epoch.
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_start,
        }

        for name, value in train_metrics.items():
            row[f"train_{name}"] = value

        for name, value in val_metrics.items():
            row[f"val_{name}"] = value

        history.append(row)

        # Save history after every epoch.
        pd.DataFrame(history).to_csv(
            out / "training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.3f} | "
            f"Val recall: {val_metrics['recall']:.3f} | "
            f"Val F1: {val_metrics['f1']:.3f} | "
            f"Time: {row['seconds']:.1f}s",
            flush=True,
        )

        # Save the model whenever validation loss improves.
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "config": config,
                },
                out / "best_model.pt",
            )

        else:
            epochs_without_improvement += 1

        # Stop after several epochs without validation loss improvement.
        if epochs_without_improvement >= PATIENCE:
            print(
                "Early stopping: validation loss stopped improving."
            )
            break

    # Generate training and validation learning curves.
    history_table = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        history_table["epoch"],
        history_table["train_loss"],
        label="Training",
    )
    axes[0].plot(
        history_table["epoch"],
        history_table["val_loss"],
        label="Validation",
    )
    axes[0].set_title("Weighted loss")
    axes[0].set_ylabel("Loss")

    axes[1].plot(
        history_table["epoch"],
        history_table["train_accuracy"],
        label="Training",
    )
    axes[1].plot(
        history_table["epoch"],
        history_table["val_accuracy"],
        label="Validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()

    plt.tight_layout()
    plt.savefig(out / "learning_curves.png", dpi=200)
    plt.close(fig)

    # Print the final experiment summary.
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(
        f"Total training time: "
        f"{(time.perf_counter() - training_start) / 60:.1f} minutes"
    )
    print(f"Results saved to: {out}")


# Start training when this file is executed directly.
if __name__ == "__main__":
    main()