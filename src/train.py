import argparse
import os
import sys

import torch
from torch_geometric.loader import DataLoader

from model import GCN
from dataset import MoleculeDataset


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
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y.view(-1)).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


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
    
    model = GCN(hidden_channels=args.hidden_channels, data=train_dataset)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
# Example usage:
# python src/train.py --epochs 500 --hidden-channels 64 --dataset_path data/ --learning-rate 0.01