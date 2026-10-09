"""
select_thresholds.py

Select validation thresholds for:
    InitialCNN
    Square-trained ResNet-50
    Letterbox-trained ResNet-50

Selection rule:
    1. Highest balanced accuracy.
    2. If tied, higher specificity.
    3. If tied, threshold closest to 0.5.
    4. If still tied, lower threshold.

Balanced accuracy:
    (pneumonia recall + specificity) / 2

Outputs:
    results/threshold_selection/<timestamp>/
        threshold_comparison.csv
        <model>_threshold_candidates.csv
        selected_thresholds.json
        summary.txt

Run:
    .\\.venv\\Scripts\\python.exe .\\select_thresholds.py

No retraining or image inference is performed.
The test set is not read or evaluated.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gradcam_resnet50 import get_validation_hash, sha256_file


ROOT = Path(__file__).resolve().parent

# Pin the exact completed runs so later training does not change selection.
MODELS = {
    "InitialCNN": {
        "run": ROOT / "results" / "initial" / "20261006_222400_613495",
        "preprocessing": "grayscale_square_resize",
    },
    "ResNet50_square": {
        "run": ROOT / "results" / "resnet50" / "20261007_172849_403643",
        "preprocessing": "square_resize_imagenet_normalization",
    },
    "ResNet50_letterbox": {
        "run": (
            ROOT / "results" / "resnet50_letterbox"
            / "20261007_213723_403333"
        ),
        "preprocessing": (
            "aspect_preserving_black_letterbox_imagenet_normalization"
        ),
    },
}


def calculate_metrics(targets, scores, threshold):
    """Calculate metrics for a single operating threshold."""
    predicted = (scores >= threshold).astype(int)

    tn = int(np.sum((targets == 0) & (predicted == 0)))
    fp = int(np.sum((targets == 0) & (predicted == 1)))
    fn = int(np.sum((targets == 1) & (predicted == 0)))
    tp = int(np.sum((targets == 1) & (predicted == 1)))

    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(targets),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def load_validation_results(model_name, run):
    """Verify the checkpoint and align its saved validation predictions."""
    checkpoint_path = run / "best_model.pt"
    validation_path = run / "val.csv"
    prediction_path = run / "validation_evaluation" / "predictions.csv"

    for path in [checkpoint_path, validation_path, prediction_path]:
        if not path.exists():
            raise SystemExit(f"Required file not found:\n{path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    # Confirm that the saved assignments match those recorded at training.
    if sha256_file(validation_path) != get_validation_hash(config):
        raise SystemExit(
            f"{model_name}: validation assignments have changed."
        )

    validation = pd.read_csv(validation_path)
    predictions = pd.read_csv(prediction_path)

    if not validation["path"].is_unique:
        raise SystemExit(f"{model_name}: duplicate validation paths.")

    if not predictions["path"].is_unique:
        raise SystemExit(f"{model_name}: duplicate prediction paths.")

    if set(validation["path"]) != set(predictions["path"]):
        raise SystemExit(
            f"{model_name}: prediction and validation image paths differ."
        )

    # Align predictions by image identity rather than assuming row order.
    predictions = (
        predictions.set_index("path")
        .loc[validation["path"]]
        .reset_index()
    )

    targets = validation["target"].to_numpy(dtype=int)
    scores = predictions["pneumonia_score"].to_numpy(dtype=float)

    if set(targets) != {0, 1}:
        raise SystemExit(f"{model_name}: validation needs both classes.")

    if not np.array_equal(
        targets,
        predictions["target"].to_numpy(dtype=int),
    ):
        raise SystemExit(f"{model_name}: validation labels differ.")

    if not np.isfinite(scores).all():
        raise SystemExit(f"{model_name}: non-finite scores found.")

    if np.any((scores < 0) | (scores > 1)):
        raise SystemExit(f"{model_name}: scores must be between 0 and 1.")

    # Verify that the saved classifications agree with the saved scores.
    saved_threshold = float(config["decision_threshold"])
    expected_predictions = (scores >= saved_threshold).astype(int)

    if not np.array_equal(
        expected_predictions,
        predictions["predicted_target"].to_numpy(dtype=int),
    ):
        raise SystemExit(
            f"{model_name}: saved scores and classifications disagree."
        )

    return checkpoint, validation, targets, scores


def select_threshold(targets, scores):
    """Search observed score thresholds and apply the shared ranking rule."""
    # At observed scores, the >= rule captures each available decision boundary.
    # Include 0.5 explicitly for comparison with earlier evaluations.
    thresholds = np.unique(
        np.concatenate([
            scores,
            np.array([0.0, 0.5, 1.0]),
        ])
    )

    candidates = pd.DataFrame([
        calculate_metrics(targets, scores, threshold)
        for threshold in thresholds
    ])

    # Temporary columns make tie handling explicit and reproducible.
    candidates["_balanced_rank"] = (
        candidates["balanced_accuracy"].round(12)
    )
    candidates["_specificity_rank"] = (
        candidates["specificity"].round(12)
    )
    candidates["_distance_from_half"] = (
        candidates["threshold"] - 0.5
    ).abs()

    ranked = candidates.sort_values(
        by=[
            "_balanced_rank",
            "_specificity_rank",
            "_distance_from_half",
            "threshold",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    )

    selected_threshold = float(ranked.iloc[0]["threshold"])

    candidates = candidates.drop(columns=[
        "_balanced_rank",
        "_specificity_rank",
        "_distance_from_half",
    ])

    return selected_threshold, candidates


def main():
    comparison_rows = []
    selections = {}
    candidate_tables = {}
    reference_assignments = None

    # --------------------------------------------------------
    # Select thresholds using validation predictions only
    # --------------------------------------------------------

    for model_name, settings in MODELS.items():
        run = settings["run"]
        print(f"Selecting threshold for {model_name}...")

        checkpoint, validation, targets, scores = (
            load_validation_results(model_name, run)
        )

        # Verify that all models use identical validation images and labels.
        assignments = (
            validation[["path", "target"]]
            .sort_values("path")
            .reset_index(drop=True)
        )

        if reference_assignments is None:
            reference_assignments = assignments
        elif not assignments.equals(reference_assignments):
            raise SystemExit(
                "The models do not use identical validation assignments."
            )

        # Check the letterbox metadata before recording its preprocessing.
        if model_name == "ResNet50_letterbox":
            if checkpoint["config"].get("preprocessing") != (
                "aspect_preserving_black_letterbox"
            ):
                raise SystemExit(
                    "Letterbox checkpoint preprocessing metadata differs."
                )

        threshold, candidates = select_threshold(targets, scores)
        candidate_tables[model_name] = candidates

        for setting, operating_threshold in [
            ("fixed_0.5", 0.5),
            ("validation_selected", threshold),
        ]:
            comparison_rows.append({
                "model": model_name,
                "setting": setting,
                **calculate_metrics(
                    targets,
                    scores,
                    operating_threshold,
                ),
            })

        checkpoint_path = run / "best_model.pt"

        # Preserve full precision and exact checkpoint identities.
        selections[model_name] = {
            "run": str(run.relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_csv_sha256": sha256_file(run / "val.csv"),
            "validation_predictions_sha256": sha256_file(
                run / "validation_evaluation" / "predictions.csv"
            ),
            "validation_images": len(validation),
            "threshold": threshold,
            "reference_threshold": 0.5,
            "preprocessing": settings["preprocessing"],
        }

    # --------------------------------------------------------
    # Save results after all validation checks pass
    # --------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "results" / "threshold_selection" / timestamp
    output.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output / "threshold_comparison.csv",
        index=False,
    )

    for model_name, candidates in candidate_tables.items():
        candidates.to_csv(
            output / f"{model_name}_threshold_candidates.csv",
            index=False,
        )

    protocol = {
        "selection_dataset": "validation",
        "selection_rule": "maximize balanced accuracy",
        "tie_breakers": [
            "higher specificity",
            "threshold closest to 0.5",
            "lower threshold",
        ],
        "prediction_rule": "pneumonia_score >= threshold",
        "test_set_used": False,
        "models": selections,
    }

    (output / "selected_thresholds.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "model",
        "setting",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "recall",
        "specificity",
        "false_negatives",
        "false_positives",
    ]

    displayed = comparison[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )

    summary = "\n".join([
        "Threshold selection used validation predictions only.",
        "Rule: maximize balanced accuracy.",
        "Balanced accuracy = (pneumonia recall + specificity) / 2.",
        "",
        displayed,
        "",
        "Exact thresholds are saved in selected_thresholds.json.",
        "Each model retains its matching preprocessing.",
        "Validation was also used for checkpoint selection.",
        "These are development results, not independent test estimates.",
        "The selection rule is for research, not a clinical requirement.",
        "Scores are not calibrated clinical probabilities.",
        "Checkpoint files and saved predictions were not modified.",
        "The test set was not read or evaluated.",
    ])

    (output / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + summary)
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()