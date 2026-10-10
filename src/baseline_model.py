import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.base import clone

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
    args = parser.parse_args()
    
    train_data = pd.read_csv(os.path.join(args.data, "train_fingerprints.csv"))
    test_data = pd.read_csv(os.path.join(args.data, "test_fingerprints.csv"))

    train_labels = pd.read_csv(os.path.join(args.data, "train_labels.csv"))
    test_labels = pd.read_csv(os.path.join(args.data, "test_labels.csv"))

    estimators = [create_estimator(method="random_forest"),
                  create_estimator(method="logistic_regression"),
                  create_estimator(method="gradient_boosting")]
    
    for i, estimator in enumerate(estimators):
        clf = train(estimator, train_data.values, train_labels.values.ravel())
        evaluation_results = evaluate(clf, test_data.values, test_labels.values.ravel())
        print(f"Evaluation results for estimator {estimator.__class__.__name__}: {evaluation_results}")

        predictions = (clf.predict_proba(test_data.values)[:, 1] >= 0.5).astype(int)

        if args.confusion_matrix_path:
            os.makedirs(os.path.join(args.confusion_matrix_path, f"{estimator.__class__.__name__}"), exist_ok=True)
            save_confusion_matrix_as_csv(test_labels.values.ravel(), predictions, os.path.join(args.confusion_matrix_path, f"{estimator.__class__.__name__}/confusion_matrix.csv"))
            save_confusion_matrix_as_image(test_labels.values.ravel(), predictions, os.path.join(args.confusion_matrix_path, f"{estimator.__class__.__name__}/confusion_matrix.png"))