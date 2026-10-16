
"""Split conformal classification for molecular activity prediction.

Uses the hinge nonconformity score:
    score(x, y) = 1 - P(y | x)

The classifier must be trained before conformal calibration. Model selection
and probability calibration must not use the conformal calibration or test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibration import (
    _read_predictions,
    apply_temperature,
    check_split_filename,
    load_temperature,
)


class MolecularConformalPredictor:
    """Conformal prediction sets for a pre-trained binary classifier."""

    def __init__(
        self,
        classes: tuple[int, ...] = (0, 1),
        confidence: float = 0.90,
        seed: int = 42,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("classes must contain unique class labels.")

        self.classes = np.asarray(classes)
        self.confidence = confidence
        self.seed = seed
        self.is_fitted = False
        self.calibration_scores = np.empty(0, dtype=float)

    def _validate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)

        if probabilities.ndim != 2:
            raise ValueError("Probabilities must have shape (n_samples, n_classes).")

        if probabilities.shape[1] != len(self.classes):
            raise ValueError(
                f"Expected {len(self.classes)} probability columns; "
                f"received {probabilities.shape[1]}."
            )

        if probabilities.shape[0] == 0:
            raise ValueError("At least one sample is required.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities contain non-finite values.")

        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Probabilities must lie in [0, 1].")

        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Each row of probabilities must sum to 1.")

        return probabilities

    def _validate_labels(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("Labels must be a one-dimensional array.")

        if not np.isin(labels, self.classes).all():
            raise ValueError(f"Labels must belong to {self.classes.tolist()}.")

        return labels

    def fit(
        self,
        probabilities_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "MolecularConformalPredictor":
        """Fit the conformal predictor on held-out calibration predictions."""
        probabilities_cal = self._validate_probabilities(probabilities_cal)
        y_cal = self._validate_labels(y_cal)

        if len(probabilities_cal) != len(y_cal):
            raise ValueError("Calibration probabilities and labels must align.")

        class_indices = np.asarray([
            np.flatnonzero(self.classes == label)[0]
            for label in y_cal
        ])
        self.calibration_scores = 1.0 - probabilities_cal[
            np.arange(len(y_cal)), class_indices
        ]
        self.is_fitted = True
        self.n_calibration = len(y_cal)

        return self

    def _threshold(self, confidence: float) -> float:
        rank = int(np.ceil((len(self.calibration_scores) + 1) * confidence))
        if rank > len(self.calibration_scores):
            return 1.0
        return float(np.sort(self.calibration_scores)[rank - 1])

    def predict(
        self,
        probabilities: np.ndarray,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Return prediction sets, sizes, and class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        probabilities = self._validate_probabilities(probabilities)
        confidence = self.confidence if confidence is None else confidence

        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        threshold = self._threshold(confidence)
        set_matrix = [
            self.classes[1.0 - row <= threshold].tolist()
            for row in probabilities
        ]
        set_sizes = np.asarray([len(prediction_set) for prediction_set in set_matrix])

        return {
            "prediction_sets": set_matrix,
            "set_sizes": set_sizes,
            "confidence": confidence,
            "threshold": threshold,
            "probabilities": probabilities,
            "predicted_labels": self.classes[np.argmax(probabilities, axis=1)],
        }

    def evaluate(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        confidence: float | None = None,
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate prediction sets, including optional subgroup coverage."""
        probabilities = self._validate_probabilities(probabilities)
        y_true = self._validate_labels(y_true)

        if len(probabilities) != len(y_true):
            raise ValueError("Probabilities and labels must align.")

        result = self.predict(probabilities, confidence=confidence)
        prediction_sets = result["prediction_sets"]
        set_sizes = result["set_sizes"]

        covered = np.asarray([
            label in prediction_set
            for label, prediction_set in zip(y_true, prediction_sets)
        ])

        metrics: dict[str, Any] = {
            "n_samples": int(len(y_true)),
            "n_calibration": int(self.n_calibration),
            "confidence_target": result["confidence"],
            "nonconformity_threshold": result["threshold"],
            "empirical_coverage": float(covered.mean()),
            "coverage_error_rate": float((~covered).mean()),
            "average_set_size": float(set_sizes.mean()),
            "singleton_rate": float((set_sizes == 1).mean()),
            "empty_set_rate": float((set_sizes == 0).mean()),
            "ambiguous_set_rate": float((set_sizes > 1).mean()),
            "coverage_by_class": {},
        }

        for label in self.classes:
            mask = y_true == label
            metrics["coverage_by_class"][str(label)] = (
                {
                    "n_samples": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                }
                if mask.any()
                else {"n_samples": 0, "coverage": None}
            )

        if groups is not None:
            groups = np.asarray(groups)

            if groups.ndim != 1 or len(groups) != len(y_true):
                raise ValueError("groups must align with the evaluated samples.")
            if pd.isna(groups).any():
                raise ValueError("groups must not contain missing values.")

            metrics["coverage_by_group"] = {}

            for group in pd.unique(groups):
                mask = groups == group
                metrics["coverage_by_group"][str(group)] = {
                    "n_samples": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "average_set_size": float(set_sizes[mask].mean()),
                }

        return metrics


def _read_groups(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    groups_frame = pd.read_csv(path, header=None)
    if groups_frame.shape[1] != 1:
        raise ValueError(f"{path} must contain exactly one group column.")
    return groups_frame.iloc[:, 0].to_numpy()


def _read_probability_data(
    path: str,
    temperature_file: str | None,
    model_name: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    logits, labels = _read_predictions(path)
    temperature = (
        load_temperature(temperature_file, expected_model=model_name)
        if temperature_file
        else 1.0
    )
    positive_probabilities = apply_temperature(logits, temperature)
    probabilities = np.column_stack((1.0 - positive_probabilities, positive_probabilities))
    return probabilities, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit split conformal prediction sets on calibration-split predictions "
            "and evaluate them on test-split predictions."
        )
    )
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--temperature-file")
    parser.add_argument("--model-name")
    parser.add_argument("--output-dir", default="model/conformal")
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument(
        "--groups",
        help="Optional headerless CSV with one group value per test sample.",
    )
    parser.add_argument("--mlflow-experiment")
    parser.add_argument("--mlflow-tracking-uri")
    args = parser.parse_args()

    if Path(args.calibration_predictions).resolve() == Path(args.test_predictions).resolve():
        parser.error("--calibration-predictions and --test-predictions must be different files.")
    try:
        check_split_filename(
            args.calibration_predictions, "calibration", "--calibration-predictions"
        )
        check_split_filename(args.test_predictions, "test", "--test-predictions")
        calibration_probabilities, calibration_labels = _read_probability_data(
            args.calibration_predictions, args.temperature_file, args.model_name
        )
        test_probabilities, test_labels = _read_probability_data(
            args.test_predictions, args.temperature_file, args.model_name
        )
        groups = _read_groups(args.groups)
        conformal = MolecularConformalPredictor(confidence=args.confidence)
        conformal.fit(calibration_probabilities, calibration_labels)
        results = conformal.predict(test_probabilities)
        metrics = conformal.evaluate(
            test_probabilities, test_labels, groups=groups
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / "conformal_predictions.csv"
    pd.DataFrame({
        "label": test_labels,
        "probability_0": test_probabilities[:, 0],
        "probability_1": test_probabilities[:, 1],
        "predicted_label": results["predicted_labels"],
        "prediction_set": [
            json.dumps(prediction_set)
            for prediction_set in results["prediction_sets"]
        ],
        "set_size": results["set_sizes"],
    }).to_csv(prediction_path, index=False)

    report = {
        "model": args.model_name,
        "confidence": args.confidence,
        "calibration_predictions_file": str(args.calibration_predictions),
        "test_predictions_file": str(args.test_predictions),
        "temperature_file": args.temperature_file,
        "metrics": metrics,
    }
    report_path = output_directory / "conformal_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.mlflow_experiment:
        import mlflow

        from mlflow_utils import configure_mlflow

        configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"conformal-{args.model_name or 'custom'}"):
            mlflow.set_tags({
                "stage": "conformal",
                "model": args.model_name or "custom",
            })
            mlflow.log_params({
                "confidence": args.confidence,
                "n_calibration": metrics["n_calibration"],
                "temperature_file": args.temperature_file or "",
            })
            mlflow.log_metrics({
                name: metrics[name]
                for name in (
                    "empirical_coverage",
                    "nonconformity_threshold",
                    "coverage_error_rate",
                    "average_set_size",
                    "singleton_rate",
                    "empty_set_rate",
                    "ambiguous_set_rate",
                )
            })
            mlflow.log_artifacts(str(output_directory), artifact_path="conformal")

    print(f"Conformal metrics: {metrics}")
    print(f"Saved conformal report to: {report_path}")
    print(f"Saved prediction sets to: {prediction_path}")


if __name__ == "__main__":
    main()
    