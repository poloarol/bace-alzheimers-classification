import argparse
import copy
import os
import sys

import torch
from torch_geometric.loader import DataLoader
from tqdm import trange


from model import GCN
from dataset import MoleculeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            pred = output.argmax(dim=1)

            correct += (pred == data.y.view(-1)).sum().item()
            total += data.y.size(0)

    return correct / total if total else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Number of hidden channels in the GCN model.")
    parser.add_argument("--dataset_path", type=str, default="data/", help="Path to the dataset directory.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    
    args = parser.parse_args()
    
    train_dataset = MoleculeDataset(root=args.dataset_path, filename="bace_processed.csv")
    test_dataset = MoleculeDataset(root=args.dataset_path, filename="bace_processed.csv", test=True)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_loader = [(data.to(device)) for data in train_loader]
    test_loader = [(data.to(device)) for data in test_loader]
    
    model = GCN(hidden_channels=args.hidden_channels, data=train_dataset)
    model = model.to(device)
    
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
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)

            progress.set_postfix(
                train_acc=f"{train_acc:.4f}",
                test_acc=f"{test_acc:.4f}",
                best_val_acc=f"{best_val_acc:.4f}",
            )

            if test_acc > best_val_acc:
                best_val_acc = test_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += eval_every if epoch != 1 else 1

            if epochs_without_improvement >= patience:
                progress.write(
                    f"Early stopping at epoch {epoch}. "
                    f"Best test accuracy: {best_val_acc:.4f}"
                )
                break

    model.load_state_dict(best_model_state)
        
# Example usage:
# python src/train.py --epochs 500 --hidden-channels 64 --dataset_path data/ --learning-rate 0.01