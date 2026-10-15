import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.base import clone

from mlflow_utils import configure_mlflow, log_parameters
from calibration import (
    calibrate_and_compare,
    calibration_path_for,
    probabilities_to_logits,
    prediction_paths,
    save_predictions,
    save_temperature,
)

seed: int = 42

random.seed(seed)
np.random.seed(seed)


def create_estimator(method: str = "random_forest"):
    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
        )
    elif method == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
        )
    elif method == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")

def train(estimator: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = clone(estimator)
    clf.fit(X, y)
    return clf

def evaluate(clf: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        raise ValueError(
            "The estimator needs predict_proba or decision_function "
            "to calculate ROC-AUC and PR-AUC."
        )

    predictions = (scores >= 0.5).astype(int)

    return {
        "ROC-AUC": roc_auc_score(y, scores),
        "PR-AUC": average_precision_score(y, scores),
        "Balanced accuracy": balanced_accuracy_score(y, predictions),
        "Inhibitor F1": f1_score(y, predictions, pos_label=1),
    }


def _generate_confusion_matrix(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, predictions)

def save_confusion_matrix_as_csv(y: np.ndarray, predictions: np.ndarray, filepath: str) -> None:
    cm = _generate_confusion_matrix(y, predictions)
    pd.DataFrame(cm).to_csv(filepath, index=False)

def save_confusion_matrix_as_image(y: np.ndarray, predictions: np.ndarray, filepath: str) -> None:
    cm = _generate_confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a RandomForestClassifier.")
    parser.add_argument("--data", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--confusion-matrix-path", type=str, help="Path to save the confusion matrix as CSV and image.")
    parser.add_argument(
        "--predictions-dir",
        help="Directory for validation/test logit CSVs used by calibration. "
        "Defaults to <confusion-matrix-path>/predictions.",
    )
    parser.add_argument("--mlflow-experiment", default="bace-baseline", help="MLflow experiment name.")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or ./mlruns.")
    args = parser.parse_args()

    configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
    
    train_data = pd.read_csv(os.path.join(args.data, "train_fingerprints.csv"))
    validation_data = pd.read_csv(os.path.join(args.data, "validation_fingerprints.csv"))
    calibration_data = pd.read_csv(os.path.join(args.data, "calibration_fingerprints.csv"))
    test_data = pd.read_csv(os.path.join(args.data, "test_fingerprints.csv"))

    train_labels = pd.read_csv(os.path.join(args.data, "train_labels.csv"))
    validation_labels = pd.read_csv(os.path.join(args.data, "validation_labels.csv"))
    calibration_labels = pd.read_csv(os.path.join(args.data, "calibration_labels.csv"))
    test_labels = pd.read_csv(os.path.join(args.data, "test_labels.csv"))

    estimators = [create_estimator(method="random_forest"),
                  create_estimator(method="logistic_regression"),
                  create_estimator(method="gradient_boosting")]
    
    fitted_models = {}
    for i, estimator in enumerate(estimators):
        model_name = estimator.__class__.__name__
        with mlflow.start_run(run_name=model_name):
            log_parameters({
                "model": model_name,
                "data_dir": args.data,
                "train_samples": len(train_data),
                "validation_samples": len(validation_data),
                "calibration_samples": len(calibration_data),
                "test_samples": len(test_data),
                **estimator.get_params(),
            })
            clf = train(estimator, train_data.values, train_labels.values.ravel())
            # Model selection uses validation only; test is evaluated after selection.
            evaluation_results = {
                f"validation_{metric}": value
                for metric, value in evaluate(
                    clf,
                    validation_data.values,
                    validation_labels.values.ravel(),
                ).items()
            }
            mlflow.log_metrics(evaluation_results)
            fitted_models[model_name] = (clf, evaluation_results["validation_PR-AUC"], mlflow.active_run().info.run_id)
            mlflow.sklearn.log_model(
                sk_model=clf,
                name=model_name,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            print(f"Validation results for {model_name}: {evaluation_results}")

    best_name, (best_clf, _, best_run_id) = max(fitted_models.items(), key=lambda item: item[1][1])
    print(f"Selected model (by validation PR-AUC): {best_name}")

    with mlflow.start_run(run_id=best_run_id):
        final_results = {}
        for split_name, (features, split_labels) in {
            "calibration": (calibration_data, calibration_labels),
            "test": (test_data, test_labels),
        }.items():
            final_results.update({
                f"{split_name}_{metric}": value
                for metric, value in evaluate(best_clf, features.values, split_labels.values.ravel()).items()
            })
        mlflow.log_metrics(final_results)
        mlflow.set_tag("selected_model", "true")
        print(f"Final evaluation results for {best_name}: {final_results}")

        validation_probabilities = best_clf.predict_proba(validation_data.values)
        test_probabilities = best_clf.predict_proba(test_data.values)
        validation_logits = probabilities_to_logits(validation_probabilities)
        test_logits = probabilities_to_logits(test_probabilities)
        predictions_dir = args.predictions_dir or os.path.join(
            args.confusion_matrix_path or "model", "predictions"
        )
        validation_predictions_path, test_predictions_path = prediction_paths(predictions_dir, best_name)
        save_predictions(validation_labels.values.ravel(), validation_logits, validation_predictions_path)
        save_predictions(test_labels.values.ravel(), test_logits, test_predictions_path)
        Path(predictions_dir, "selected_baseline.json").write_text(
            json.dumps({"model": best_name}), encoding="utf-8"
        )
        mlflow.log_artifacts(os.path.dirname(validation_predictions_path), artifact_path="predictions")
        print(f"Saved calibration inputs to: {os.path.dirname(validation_predictions_path)}")

        # Fit T on validation, compare on the same test samples.
        calibration_dir = os.path.join("model", "calibration", best_name)
        calibration = calibrate_and_compare(
            validation_logits,
            validation_labels.values.ravel(),
            test_logits,
            test_labels.values.ravel(),
            calibration_dir,
        )
        save_temperature(
            calibration["temperature"],
            calibration_path_for(os.path.join("model", f"{best_name}.pkl")),
            model=best_name,
            fitted_on="validation",
            input="log of predict_proba",
        )
        mlflow.log_metric("temperature", calibration["temperature"])
        mlflow.log_metrics({
            f"test_{kind}_{metric}": value
            for kind in ("uncalibrated", "calibrated")
            for metric, value in calibration[kind].items()
        })
        mlflow.log_artifacts(calibration_dir, artifact_path="calibration")
        print(f"Temperature: {calibration['temperature']:.4f}")
        print(f"Test calibration (uncalibrated): {calibration['uncalibrated']}")
        print(f"Test calibration (temperature scaled): {calibration['calibrated']}")

        if args.confusion_matrix_path:
            predictions = (best_clf.predict_proba(test_data.values)[:, 1] >= 0.5).astype(int)
            output_dir = os.path.join(args.confusion_matrix_path, best_name)
            os.makedirs(output_dir, exist_ok=True)
            confusion_csv = os.path.join(output_dir, "confusion_matrix.csv")
            confusion_image = os.path.join(output_dir, "confusion_matrix.png")
            save_confusion_matrix_as_csv(test_labels.values.ravel(), predictions, confusion_csv)
            save_confusion_matrix_as_image(test_labels.values.ravel(), predictions, confusion_image)
            mlflow.log_artifacts(output_dir, artifact_path="confusion_matrix")

# Example usage:
# python src/baseline_model.py --data data/fingerprints --confusion-matrix-path data/confusion_matrices
# mlflow ui --backend-store-uri sqlite:///mlflow.db