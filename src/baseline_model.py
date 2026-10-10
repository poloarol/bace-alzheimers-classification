import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.base import clone

seed: int = 42

random.seed(seed)
np.random.seed(seed)


def create_estimator() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )

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
    parser.add_argument("--train-data", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--confusion-matrix-csv", type=str, help="Path to save the confusion matrix as CSV.")
    parser.add_argument("--confusion-matrix-image", type=str, help="Path to save the confusion matrix as an image.")
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)

    X_train = train_data.drop(columns=["Class"]).values
    y_train = train_data["Class"].values
    X_test = test_data.drop(columns=["Class"]).values
    y_test = test_data["Class"].values

    estimator = create_estimator()
    clf = train(estimator, X_train, y_train)
    evaluation_results = evaluate(clf, X_test, y_test)
    print(evaluation_results)

    predictions = (clf.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    if args.confusion_matrix_csv:
        save_confusion_matrix_as_csv(y_test, predictions, args.confusion_matrix_csv)
    if args.confusion_matrix_image:
        save_confusion_matrix_as_image(y_test, predictions, args.confusion_matrix_image)