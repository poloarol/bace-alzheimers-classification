import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader

from dataset import MoleculeDataset
from mlflow_utils import DEFAULT_TRACKING_URI


def _output_directory(output_root: str, model_type: str) -> Path:
    model_path = Path(model_type)
    if model_path.name != model_type or model_type in {".", ".."}:
        raise ValueError("model_type must be a single directory name.")

    output_directory = Path(output_root) / model_path
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_directory: Path,
) -> np.ndarray:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    np.savetxt(
        output_directory / "confusion_matrix.txt",
        matrix,
        fmt="%d",
        header="Rows: actual class (0, 1). Columns: predicted class (0, 1).",
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    return matrix


def run_sklearn_inference(
    model_uri: str,
    fingerprint_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    test_data_path = Path(fingerprint_dir) / "test_fingerprints.csv"
    test_labels_path = Path(fingerprint_dir) / "test_labels.csv"
    features = pd.read_csv(test_data_path)
    labels = pd.read_csv(test_labels_path).iloc[:, 0].to_numpy(dtype=int)

    model = mlflow.sklearn.load_model(model_uri)
    predictions = np.asarray(model.predict(features), dtype=int)
    return labels, predictions


def run_gnn_inference(
    model_uri: str,
    dataset_root: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = MoleculeDataset(
        root=dataset_root,
        filename="bace_processed.csv",
        test=True,
    )
    if not dataset:
        raise ValueError("The test dataset is empty.")

    model = mlflow.pytorch.load_model(model_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    labels = []
    predictions = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            labels.extend(batch.y.view(-1).cpu().tolist())

    return np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tracked model on the BACE test set."
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, such as runs:/<run-id>/model.",
    )
    parser.add_argument(
        "--model-flavor",
        choices=("sklearn", "pytorch"),
        required=True,
        help="MLflow flavor used to log the model.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        help="Single directory name used to store inference results.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/MoleculeNet/bace/output",
        help="Root directory for inference results.",
    )
    parser.add_argument(
        "--fingerprint-dir",
        help="Directory containing test_fingerprints.csv and test_labels.csv.",
    )
    parser.add_argument(
        "--dataset-root",
        help="BACE dataset root for PyG inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of graphs per PyG inference batch.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI used to resolve the model URI.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    output_directory = _output_directory(args.output_dir, args.model_type)

    if args.model_flavor == "sklearn":
        if not args.fingerprint_dir:
            parser.error("--fingerprint-dir is required for sklearn inference.")
        labels, predictions = run_sklearn_inference(
            args.model_uri,
            args.fingerprint_dir,
        )
    else:
        if not args.dataset_root:
            parser.error("--dataset-root is required for pytorch inference.")
        labels, predictions = run_gnn_inference(
            args.model_uri,
            args.dataset_root,
            args.batch_size,
        )

    matrix = save_confusion_matrix(labels, predictions, output_directory)
    accuracy = accuracy_score(labels, predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved confusion matrix to: {output_directory}")
    print(matrix)


if __name__ == "__main__":
    main()


# Example usage:
# python src/inference.py --model-uri runs:/<run-id>/model --model-flavor sklearn --model-type RandomForest --fingerprint-dir data/fingerprints --output-dir data/output
# python src/inference.py --model-uri runs:/<run-id>/model --model-flavor pytorch --model-type GCN --dataset-root data/MoleculeNet/bace --output-dir data/output