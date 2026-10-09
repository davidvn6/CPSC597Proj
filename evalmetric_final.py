"""
evalmetric_final.py

Final evaluation of:
    InitialCNN
    Square-trained ResNet-50
    Letterbox-trained ResNet-50

Uses:
    - Frozen checkpoints and validation-selected thresholds.
    - The custom test assignments in results/splits/test.csv.
    - Matching preprocessing for each model.

For each model, report:
    - Results at threshold 0.5.
    - Results at its validation-selected threshold.
    - 95% candidate-group bootstrap confidence intervals.

Inference scores are saved and reused if this script is rerun.
Thresholds are never selected from test results.

Run:
    .\\.venv\\Scripts\\python.exe .\\evalmetric_final.py

Outputs:
    results/final_test/<threshold_selection_folder>/

Candidate groups are not verified patient identities.
"""

import gc
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from train_initial import InitialCNN, XRayDataset
from train_resnet50 import ResNet50CNN
from resize_test_resnet50 import AspectPreservingResize
from gradcam_resnet50 import sha256_file


# ------------------------------------------------------------
# Project settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "results" / "splits" / "test.csv"

SEED = 42
BOOTSTRAP_REPLICATES = 2000

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
]

EXPECTED_MODELS = {
    "InitialCNN",
    "ResNet50_square",
    "ResNet50_letterbox",
}


def calculate_metrics(targets, scores, threshold, auc=None):
    """Calculate classification metrics and confusion counts."""
    predicted = (scores >= threshold).astype(int)

    tn = int(np.sum((targets == 0) & (predicted == 0)))
    fp = int(np.sum((targets == 0) & (predicted == 1)))
    fn = int(np.sum((targets == 1) & (predicted == 0)))
    tp = int(np.sum((targets == 1) & (predicted == 1)))

    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    if auc is None:
        auc = float(roc_auc_score(targets, scores))

    return {
        "accuracy": (tp + tn) / len(targets),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": recall,
        "specificity": specificity,
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn
            else 0.0
        ),
        "roc_auc": float(auc),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def find_protocol():
    """Find the latest recorded threshold-selection protocol."""
    files = sorted(
        (ROOT / "results" / "threshold_selection")
        .glob("*/selected_thresholds.json")
    )

    if not files:
        raise SystemExit("Run select_thresholds.py first.")

    return files[-1]


def verify_split_separation(test, development, description):
    """Check image, candidate-group, and identical-file separation."""
    for column in ["path", "group_id", "md5"]:
        if column not in development.columns:
            raise SystemExit(
                f"{description} is missing the {column} column."
            )

        if development[column].isna().any():
            raise SystemExit(
                f"{description} has missing {column} values."
            )

        overlap = (
            set(test[column].astype(str))
            & set(development[column].astype(str))
        )

        if overlap:
            raise SystemExit(
                f"Test overlaps {description} by {column}: "
                f"{len(overlap)} overlapping values."
            )


def build_transform(model_name, config):
    """Recreate each model's matching evaluation preprocessing."""
    size = int(config["image_size"])

    if model_name == "InitialCNN":
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

    if model_name == "ResNet50_square":
        resize_steps = [
            v2.ToImage(),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]

    elif model_name == "ResNet50_letterbox":
        if config.get("preprocessing") != (
            "aspect_preserving_black_letterbox"
        ):
            raise SystemExit("Letterbox preprocessing metadata differs.")

        resize_steps = [
            AspectPreservingResize(size, "black"),
            v2.ToDtype(torch.float32, scale=True),
        ]

    else:
        raise SystemExit(f"Unknown model: {model_name}")

    return v2.Compose([
        v2.Grayscale(num_output_channels=3),
        *resize_steps,
        v2.Normalize(
            mean=config["normalization_mean"],
            std=config["normalization_std"],
        ),
    ])


def get_scores(model_name, entry, test, device, output):
    """
    Run inference once and save scores.

    On reruns, verify and reuse cached scores from this protocol.
    """
    cache_path = output / f"{model_name}_scores.csv"

    if cache_path.exists():
        cached = pd.read_csv(cache_path)

        if not cached["path"].is_unique:
            raise SystemExit(f"Duplicate cached paths for {model_name}.")

        if set(cached["path"]) != set(test["path"]):
            raise SystemExit(f"Cached test images differ for {model_name}.")

        cached = (
            cached.set_index("path")
            .loc[test["path"]]
            .reset_index()
        )

        if not np.array_equal(
            cached["target"].to_numpy(dtype=int),
            test["target"].to_numpy(dtype=int),
        ):
            raise SystemExit(f"Cached test labels differ for {model_name}.")

        print(f"Reusing saved test scores: {model_name}")
        scores = cached["pneumonia_score"].to_numpy(dtype=float)

    else:
        checkpoint = torch.load(
            ROOT / entry["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
        config = checkpoint["config"]

        if model_name == "InitialCNN":
            model = InitialCNN()
        else:
            model = ResNet50CNN(pretrained=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        loader = DataLoader(
            XRayDataset(test, build_transform(model_name, config)),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=0,
        )

        print(
            f"Evaluating {model_name}, epoch {entry['checkpoint_epoch']}, "
            f"on {len(test)} test images using {device}."
        )

        batches = []

        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device, dtype=torch.float32)
                logits = model(images).reshape(-1)
                batches.append(
                    torch.sigmoid(logits).cpu().numpy()
                )

        scores = np.concatenate(batches)

        if len(scores) != len(test) or not np.isfinite(scores).all():
            raise SystemExit(f"Invalid test scores for {model_name}.")

        cached = test.copy()
        cached["pneumonia_score"] = scores
        cached.to_csv(cache_path, index=False)

        del model, checkpoint, loader
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if (
        len(scores) != len(test)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise SystemExit(f"Invalid saved scores for {model_name}.")

    return scores


def bootstrap_intervals(test, model_scores, settings):
    """
    Resample entire candidate groups with replacement.

    The same sampled groups are used for all models in each replicate.
    Thresholds remain fixed throughout bootstrap calculation.
    """
    targets = test["target"].to_numpy(dtype=int)

    # Each array contains all image positions belonging to one group.
    group_indices = [
        np.asarray(indices, dtype=int)
        for indices in test.groupby("group_id", sort=True).indices.values()
    ]

    generator = np.random.default_rng(SEED)
    samples = {
        (name, setting): []
        for name in model_scores
        for setting in settings[name]
    }

    successful = 0

    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_groups = generator.integers(
            0,
            len(group_indices),
            size=len(group_indices),
        )

        indices = np.concatenate([
            group_indices[index] for index in sampled_groups
        ])
        sampled_targets = targets[indices]

        # ROC-AUC and class recall require both classes.
        if len(np.unique(sampled_targets)) < 2:
            continue

        for name, scores in model_scores.items():
            sampled_scores = scores[indices]
            auc = float(
                roc_auc_score(sampled_targets, sampled_scores)
            )

            for setting, threshold in settings[name].items():
                metrics = calculate_metrics(
                    sampled_targets,
                    sampled_scores,
                    threshold,
                    auc=auc,
                )

                samples[(name, setting)].append([
                    metrics[metric] for metric in METRIC_NAMES
                ])

        successful += 1

        if (replicate + 1) % 500 == 0:
            print(
                f"Bootstrap: {replicate + 1}/"
                f"{BOOTSTRAP_REPLICATES}"
            )

    if successful < int(0.95 * BOOTSTRAP_REPLICATES):
        raise SystemExit(
            "Too many bootstrap samples lacked both classes."
        )

    intervals = {}

    for key, values in samples.items():
        array = np.asarray(values)

        intervals[key] = {
            metric: (
                float(np.percentile(array[:, index], 2.5)),
                float(np.percentile(array[:, index], 97.5)),
            )
            for index, metric in enumerate(METRIC_NAMES)
        }

    return intervals, successful


def main():
    # --------------------------------------------------------
    # Load the frozen validation-selection record
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    protocol_path = find_protocol()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    if set(protocol["models"]) != EXPECTED_MODELS:
        raise SystemExit(
            "The threshold protocol must contain all three models."
        )

    if protocol.get("test_set_used") is not False:
        raise SystemExit("The protocol does not identify validation-only selection.")

    if not TEST_PATH.exists():
        raise SystemExit(f"Test assignments not found:\n{TEST_PATH}")

    test = pd.read_csv(TEST_PATH).reset_index(drop=True)

    for column in ["path", "target", "group_id", "md5"]:
        if column not in test.columns or test[column].isna().any():
            raise SystemExit(f"Missing test metadata: {column}")

    if not test["path"].is_unique or not test["md5"].is_unique:
        raise SystemExit("Duplicate test image paths or file hashes.")

    if set(test["target"].astype(int)) != {0, 1}:
        raise SystemExit("The test set must contain both classes.")

    if "split" in test.columns and not test["split"].eq("test").all():
        raise SystemExit("The test CSV contains non-test assignments.")

    # --------------------------------------------------------
    # Verify checkpoints and separation before inference
    # --------------------------------------------------------

    for name, entry in protocol["models"].items():
        checkpoint_path = ROOT / entry["checkpoint"]
        run = ROOT / entry["run"]

        if sha256_file(checkpoint_path) != entry["checkpoint_sha256"]:
            raise SystemExit(f"{name}: checkpoint has changed.")

        if sha256_file(run / "val.csv") != entry["validation_csv_sha256"]:
            raise SystemExit(f"{name}: validation assignments have changed.")

        prediction_path = (
            run / "validation_evaluation" / "predictions.csv"
        )
        if sha256_file(prediction_path) != (
            entry["validation_predictions_sha256"]
        ):
            raise SystemExit(f"{name}: validation predictions have changed.")

        for split in ["train", "val"]:
            development = pd.read_csv(run / f"{split}.csv")
            verify_split_separation(
                test,
                development,
                f"{name} {split}",
            )

    for relative_path in test["path"]:
        if not (ROOT / relative_path).exists():
            raise SystemExit(f"Test image not found:\n{ROOT / relative_path}")

    # Bind cached results to this exact protocol and test assignment file.
    output = ROOT / "results" / "final_test" / protocol_path.parent.name
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "test_csv_sha256": sha256_file(TEST_PATH),
        "test_images": len(test),
        "candidate_groups": int(test["group_id"].nunique()),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }

    manifest_path = output / "experiment_manifest.json"

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if previous != manifest:
            raise SystemExit(
                "Existing final results use a different protocol or test CSV."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    print(f"Threshold protocol: {protocol_path}")
    print(
        f"Test images: {len(test)} | "
        f"Candidate groups: {test['group_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Collect one score per test image per model
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_scores = {}
    settings = {}

    for name, entry in protocol["models"].items():
        model_scores[name] = get_scores(
            name,
            entry,
            test,
            device,
            output,
        )
        settings[name] = {
            "fixed_0.5": 0.5,
            "validation_selected": float(entry["threshold"]),
        }

    # --------------------------------------------------------
    # Calculate confidence intervals with fixed thresholds
    # --------------------------------------------------------

    intervals, successful = bootstrap_intervals(
        test,
        model_scores,
        settings,
    )

    targets = test["target"].to_numpy(dtype=int)
    metric_rows = []
    interval_rows = []
    subtype_rows = []

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

    for name, scores in model_scores.items():
        auc = float(roc_auc_score(targets, scores))
        fpr, tpr, _ = roc_curve(targets, scores)
        ax_roc.plot(fpr, tpr, label=f"{name}: {auc:.4f}")

        for setting, threshold in settings[name].items():
            predicted = (scores >= threshold).astype(int)
            metrics = calculate_metrics(
                targets,
                scores,
                threshold,
                auc=auc,
            )

            metric_rows.append({
                "model": name,
                "setting": setting,
                "threshold": threshold,
                **metrics,
            })

            for metric in METRIC_NAMES:
                lower, upper = intervals[(name, setting)][metric]

                interval_rows.append({
                    "model": name,
                    "setting": setting,
                    "metric": metric,
                    "estimate": metrics[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                })

            # Save classifications at both predefined operating points.
            predictions = test.copy()
            predictions["pneumonia_score"] = scores
            predictions["predicted_target"] = predicted
            predictions["predicted_label"] = np.where(
                predicted == 1, "PNEUMONIA", "NORMAL"
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

            predictions.to_csv(
                output / f"{name}_{setting}_predictions.csv",
                index=False,
            )

            # Save each operating point's confusion matrix.
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                confusion_matrix(targets, predicted, labels=[0, 1]),
                display_labels=["NORMAL", "PNEUMONIA"],
            ).plot(ax=ax, cmap="Blues", colorbar=False)

            ax.set_title(f"{name}\n{setting}")
            fig.tight_layout()
            fig.savefig(
                output / f"{name}_{setting}_confusion_matrix.png",
                dpi=200,
            )
            plt.close(fig)

            # Report filename-derived subtype recall descriptively.
            if "ptype" in test.columns:
                for subtype in ["bacterial", "viral"]:
                    mask = (targets == 1) & test["ptype"].eq(
                        subtype
                    ).to_numpy()

                    number = int(mask.sum())

                    if number:
                        missed = int(np.sum(predicted[mask] == 0))
                        subtype_rows.append({
                            "model": name,
                            "setting": setting,
                            "subtype": subtype,
                            "images": number,
                            "false_negatives": missed,
                            "recall": 1 - missed / number,
                        })

    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("Final test ROC curves")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(output / "roc_comparison.png", dpi=200)
    plt.close(fig_roc)

    # --------------------------------------------------------
    # Save and print final results
    # --------------------------------------------------------

    metrics_table = pd.DataFrame(metric_rows)
    ci_table = pd.DataFrame(interval_rows)

    metrics_table.to_csv(output / "metrics.csv", index=False)
    ci_table.to_csv(output / "confidence_intervals.csv", index=False)
    pd.DataFrame(subtype_rows).to_csv(
        output / "subtype_results.csv",
        index=False,
    )

    displayed = metrics_table[[
        "model",
        "setting",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "false_negatives",
        "false_positives",
    ]].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )

    summary = "\n".join([
        f"Test images: {len(test)}",
        f"Candidate groups: {test['group_id'].nunique()}",
        f"Successful bootstrap replicates: {successful}",
        "",
        displayed,
        "",
        "95% percentile intervals are saved in confidence_intervals.csv.",
        "Bootstrap sampling resampled entire candidate groups.",
        "Candidate groups are not verified patient identities.",
        "Intervals are conditional on the fitted models and this dataset.",
        "They do not include variability from retraining or threshold selection.",
        "Subtype labels come from filenames.",
        "The thresholds were selected on validation, not test.",
        "Scores are not calibrated clinical probabilities.",
        "No checkpoint, preprocessing, or threshold was changed using test results.",
    ])

    (output / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + summary)
    print(f"\nSaved final results to: {output}")


if __name__ == "__main__":
    main()