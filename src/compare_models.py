"""Compare the Morgan baseline and graph models on identical splits.

Reads the artifacts written by the baseline, GNN, calibration and conformal
stages, verifies that every model was scored on the same test molecules, and
logs one comparison run (configuration, metrics, artifacts) to MLflow.
Nothing is trained or tuned here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from audit_splits import SPLITS, build_report
from calibration import _read_predictions
from dataset import split_file_signature

BEST_BASELINE = "best-baseline"
DEFAULT_MODELS = (BEST_BASELINE, "GCN", "GAT", "GATGraphTransformer")


def _resolve_model_name(predictions_directory: Path, model: str) -> str:
    if model != BEST_BASELINE:
        return model
    selection = predictions_directory / "selected_baseline.json"
    if not selection.is_file():
        raise FileNotFoundError(f"Baseline selection file does not exist: {selection}")
    return json.loads(selection.read_text(encoding="utf-8"))["model"]


def _scores(predictions_file: Path) -> tuple[np.ndarray, np.ndarray]:
    logits, labels = _read_predictions(str(predictions_file))
    logits = np.asarray(logits, dtype=float)
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    return np.asarray(labels).astype(int), probabilities


def _classification_metrics(prefix: str, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    positive = probabilities[:, 1]
    return {
        f"{prefix}_pr_auc": float(average_precision_score(labels, positive)),
        f"{prefix}_roc_auc": float(roc_auc_score(labels, positive)),
        f"{prefix}_accuracy": float(accuracy_score(labels, predictions)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        f"{prefix}_inhibitor_f1": float(f1_score(labels, predictions, pos_label=1)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_models(dataset_root: Path, models: tuple[str, ...]) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    predictions_directory = dataset_root / "output" / "predictions"
    split_path = dataset_root / "clean" / "train_test_split.npz"
    audit = build_report(dataset_root)
    partitions = audit["partitions"]
    if not partitions["partitions_disjoint"]:
        raise ValueError("Split partitions overlap; refusing to compare models.")

    frame = pd.read_csv(dataset_root / "raw" / "bace_processed.csv")
    with np.load(split_path) as archive:
        expected_labels = {
            split: frame["Class"].to_numpy()[archive[f"{split}_idx"]] for split in SPLITS
        }

    rows: list[dict[str, Any]] = []
    for requested in models:
        name = _resolve_model_name(predictions_directory, requested)
        model_directory = predictions_directory / name
        row: dict[str, Any] = {"model": name, "role": "baseline" if requested == BEST_BASELINE else "graph"}

        for split in ("validation", "calibration", "test"):
            labels, probabilities = _scores(model_directory / f"{split}_predictions.csv")
            if not np.array_equal(labels, expected_labels[split]):
                raise ValueError(
                    f"{name} {split} labels differ from the shared split; "
                    "models were not scored on the same molecules."
                )
            if split == "validation":
                row.update(_classification_metrics("validation", labels, probabilities))
            elif split == "test":
                row.update(_classification_metrics("test", labels, probabilities))

        calibration = _load_json(dataset_root / "output" / "calibration" / name / "calibration_report.json")
        conformal = _load_json(dataset_root / "output" / "conformal" / name / "conformal_report.json")
        calibrated = calibration["metrics"]["calibrated"]
        uncalibrated = calibration["metrics"]["uncalibrated"]
        conformal_metrics = conformal["metrics"]
        row.update({
            "temperature": calibration["temperature"],
            "test_nll_uncalibrated": uncalibrated["NLL"],
            "test_nll_calibrated": calibrated["NLL"],
            "test_ece_uncalibrated": uncalibrated["ECE"],
            "test_ece_calibrated": calibrated["ECE"],
            "test_brier_calibrated": calibrated["Brier"],
            "conformal_confidence_target": conformal_metrics["confidence_target"],
            "conformal_coverage": conformal_metrics["empirical_coverage"],
            "conformal_coverage_class_0": conformal_metrics["coverage_by_class"]["0"]["coverage"],
            "conformal_coverage_class_1": conformal_metrics["coverage_by_class"]["1"]["coverage"],
            "conformal_average_set_size": conformal_metrics["average_set_size"],
            "conformal_singleton_rate": conformal_metrics["singleton_rate"],
            "conformal_n_calibration": conformal_metrics["n_calibration"],
        })
        rows.append(row)

    return {
        "split_signature": split_file_signature(split_path),
        "audit": audit,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on identical splits and log to MLflow.")
    parser.add_argument("--dataset-path", default="data/MoleculeNet/bace")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", help="Defaults to <dataset-path>/output/comparison.")
    parser.add_argument("--mlflow-experiment", default="bace-comparison")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--confidence", type=float, default=0.90)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)
    comparison = compare_models(dataset_root, tuple(args.models))
    output_directory = Path(args.output_dir or dataset_root / "output" / "comparison")
    output_directory.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(comparison["results"])
    table.to_csv(output_directory / "comparison.csv", index=False)
    configuration = {
        "dataset_path": str(dataset_root),
        "split_signature": comparison["split_signature"],
        "models": list(args.models),
        "checkpoint_selection": "validation PR-AUC (all models)",
        "probability_calibration": "temperature scaling fitted on validation split",
        "conformal_method": "split conformal (crepes), fitted on calibration split, reported on test split",
        "conformal_confidence": args.confidence,
        "epochs": args.epochs,
        "hidden_channels": args.hidden_channels,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "partition_sizes": {
            name: info["molecules"] for name, info in comparison["audit"]["partitions"]["splits"].items()
        },
    }
    (output_directory / "comparison.json").write_text(
        json.dumps({"configuration": configuration, **comparison}, indent=2, default=str),
        encoding="utf-8",
    )

    display = table[[
        "model", "validation_pr_auc", "test_pr_auc", "test_roc_auc", "test_balanced_accuracy",
        "test_inhibitor_f1", "test_ece_calibrated", "conformal_coverage", "conformal_average_set_size",
    ]]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    if not args.no_mlflow:
        import mlflow

        from mlflow_utils import configure_mlflow, log_parameters

        configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
        with mlflow.start_run(run_name="model-comparison"):
            mlflow.set_tag("stage", "comparison")
            log_parameters({
                key: value for key, value in configuration.items()
                if key != "partition_sizes" and value is not None
            })
            for split, info in comparison["audit"]["partitions"]["splits"].items():
                mlflow.log_params({
                    f"{split}_molecules": info["molecules"],
                    f"{split}_class_0": info["class_counts"]["0"],
                    f"{split}_class_1": info["class_counts"]["1"],
                    f"{split}_graphs": info["processed_graphs"],
                })
            for row in comparison["results"]:
                mlflow.log_metrics({
                    f"{row['model']}_{key}": float(value)
                    for key, value in row.items()
                    if key not in ("model", "role") and value is not None
                })
            mlflow.log_artifacts(str(output_directory), artifact_path="comparison")
            mlflow.log_artifact(str(dataset_root / "clean" / "train_test_split.npz"), artifact_path="split")

    print(f"Saved comparison to: {output_directory}")


if __name__ == "__main__":
    main()

# Example usage:
# python src/compare_models.py --dataset-path data/MoleculeNet/bace --epochs 100 --hidden-channels 64
