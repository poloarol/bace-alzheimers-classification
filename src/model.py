import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels: int, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, data.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GCN model for graph classification")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels in the GCN")
    parser.add_argument("--dataset_path", type=str, default="data/", help="Path to the dataset")
    args = parser.parse_args()

    model = GCN(hidden_channels=args.hidden_channels, data=data)
    print(model)
    
# Example usage:
# python src/model.py --hidden_channels 128 --dataset_path data/