import argparse
import copy
import os
import sys

import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader
from tqdm import trange


from model import build_model
from dataset import MoleculeDataset
from mlflow_utils import configure_mlflow, log_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch, data.edge_attr)
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def classification_metrics(labels, scores, predictions):
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "ROC-AUC": roc_auc_score(labels, scores),
        "PR-AUC": average_precision_score(labels, scores),
        "Balanced accuracy": balanced_accuracy_score(labels, predictions),
        "Inhibitor F1": f1_score(labels, predictions, pos_label=1),
    }


def evaluate(model, loader):
    model.eval()
    labels = []
    scores = []
    predictions = []

    with torch.inference_mode():
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch, data.edge_attr)
            pred = output.argmax(dim=1)

            labels.extend(data.y.view(-1).cpu().tolist())
            scores.extend(torch.softmax(output, dim=1)[:, 1].cpu().tolist())
            predictions.extend(pred.cpu().tolist())

    if not labels:
        raise ValueError("Cannot evaluate an empty data loader.")

    return classification_metrics(labels, scores, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a graph model.")
    parser.add_argument(
        "--model",
        choices=("GCN", "GAT", "GATGraphTransformer"),
        default="GCN",
        help="Graph model architecture.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Number of hidden channels in the graph model.")
    parser.add_argument("--dataset_path", type=str, default="data/", help="Path to the dataset directory.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of graphs per batch.")
    parser.add_argument(
        "--model-output",
        default=None,
        help="Optional path for the local model state-dict checkpoint.",
    )
    parser.add_argument("--mlflow-experiment", default="bace-gnn", help="MLflow experiment name.")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or ./mlruns.")
    
    args = parser.parse_args()

    configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
    mlflow.start_run(run_name=args.model)
    log_parameters({
        "dataset_path": args.dataset_path,
        "epochs": args.epochs,
        "hidden_channels": args.hidden_channels,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "model": args.model,
    })
    
    train_dataset = MoleculeDataset(
        root=args.dataset_path,
        filename="bace_processed.csv",
        split="train",
    )
    validation_dataset = MoleculeDataset(
        root=args.dataset_path,
        filename="bace_processed.csv",
        split="validation",
    )
    calibration_dataset = MoleculeDataset(
        root=args.dataset_path,
        filename="bace_processed.csv",
        split="calibration",
    )
    test_dataset = MoleculeDataset(
        root=args.dataset_path,
        filename="bace_processed.csv",
        split="test",
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(validation_dataset)}")
    print(f"Number of calibration samples: {len(calibration_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_loader = [(data.to(device)) for data in train_loader]
    validation_loader = [(data.to(device)) for data in validation_loader]
    calibration_loader = [(data.to(device)) for data in calibration_loader]
    test_loader = [(data.to(device)) for data in test_loader]
    
    model = build_model(
        model_type=args.model,
        hidden_channels=args.hidden_channels,
        data=train_dataset,
    )
    model = model.to(device)
    model_class_name = model.__class__.__name__
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    patience = 20       # Stop after 10 epochs without improvement
    eval_every = 5
    best_val_acc = 0.0
    epochs_without_improvement = 5
    best_model_state = copy.deepcopy(model.state_dict())

    progress = trange(1, args.epochs + 1, desc="Training")

    for epoch in progress:
        train(model, train_loader)

        if epoch % eval_every == 0 or epoch == 1:
            train_metrics = evaluate(model, train_loader)
            validation_metrics = evaluate(model, validation_loader)

            mlflow.log_metrics(
                {
                    "train_accuracy": train_metrics["Accuracy"],
                    "validation_accuracy": validation_metrics["Accuracy"],
                    "validation_roc_auc": validation_metrics["ROC-AUC"],
                    "validation_pr_auc": validation_metrics["PR-AUC"],
                    "validation_balanced_accuracy": validation_metrics[
                        "Balanced accuracy"
                    ],
                    "validation_inhibitor_f1": validation_metrics[
                        "Inhibitor F1"
                    ],
                },
                step=epoch,
            )

            progress.set_postfix(
                train_acc=f"{train_metrics['Accuracy']:.4f}",
                val_acc=f"{validation_metrics['Accuracy']:.4f}",
                best_val_acc=f"{best_val_acc:.4f}",
            )

            if validation_metrics["Accuracy"] > best_val_acc:
                best_val_acc = validation_metrics["Accuracy"]
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                progress.write(
                    f"Early stopping at epoch {epoch}. "
                    f"Best validation accuracy: {best_val_acc:.4f}"
                )
                break

    model.load_state_dict(best_model_state)
    mlflow.log_metric("best_validation_accuracy", best_val_acc)
    final_calibration_metrics = evaluate(model, calibration_loader)
    final_test_metrics = evaluate(model, test_loader)
    mlflow.log_metrics({
        f"calibration_{metric}": value
        for metric, value in final_calibration_metrics.items()
    })
    mlflow.log_metrics(final_test_metrics)
    print(f"Final calibration metrics: {final_calibration_metrics}")
    print(f"Final test metrics: {final_test_metrics}")
    mlflow.pytorch.log_model(
        model,
        f"{args.model}-Model",
        serialization_format="pickle",
    )
    mlflow.end_run()
    
    model_output = os.path.abspath(
        args.model_output or os.path.join("model", f"{model_class_name}.pth")
    )
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    torch.save(model.state_dict(), model_output)
        
# Example usage:
# python src/train.py --epochs 500 --hidden-channels 64 --dataset_path data/ --learning-rate 0.01


# python src/train.py --dataset_path data/MoleculeNet/bace
# mlflow ui --backend-store-uri sqlite:///mlflow.db