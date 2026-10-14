import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import GATConv, GCNConv, TransformerConv
from torch_geometric.nn import global_mean_pool

from dataset import MoleculeDataset

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

class GraphClassifier(torch.nn.Module):
    def __init__(self, hidden_channels: int, data):
        super().__init__()
        self.lin = Linear(hidden_channels, data.num_classes)

    def classify(self, x, batch):
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class GCN(GraphClassifier):
    def __init__(self, hidden_channels: int, data):
        super().__init__(hidden_channels, data)
        self.conv1 = GCNConv(data.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        return self.classify(x, batch)


class GAT(GraphClassifier):
    def __init__(self, hidden_channels: int, data, heads: int = 4):
        super().__init__(hidden_channels, data)
        edge_dim = data.num_edge_features
        self.conv1 = GATConv(
            data.num_node_features,
            hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            edge_dim=edge_dim,
        )
        self.conv3 = GATConv(
            hidden_channels,
            hidden_channels,
            edge_dim=edge_dim,
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is None:
            raise ValueError("GAT requires edge attributes.")
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr)
        return self.classify(x, batch)


class GATGraphTransformer(GraphClassifier):
    def __init__(self, hidden_channels: int, data, gat_heads: int = 4):
        super().__init__(hidden_channels, data)
        edge_dim = data.num_edge_features
        self.gat = GATConv(
            data.num_node_features,
            hidden_channels,
            heads=gat_heads,
            edge_dim=edge_dim,
        )
        gat_channels = hidden_channels * gat_heads
        self.transformer1 = TransformerConv(
            gat_channels,
            hidden_channels,
            heads=2,
            concat=False,
            edge_dim=edge_dim,
            beta=True,
        )
        self.transformer2 = TransformerConv(
            hidden_channels,
            hidden_channels,
            heads=2,
            concat=False,
            edge_dim=edge_dim,
            beta=True,
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is None:
            raise ValueError("GATGraphTransformer requires edge attributes.")
        x = self.gat(x, edge_index, edge_attr).relu()
        x = self.transformer1(x, edge_index, edge_attr).relu()
        x = self.transformer2(x, edge_index, edge_attr)
        return self.classify(x, batch)


def build_model(model_type: str, hidden_channels: int, data):
    models = {
        "GCN": GCN,
        "GAT": GAT,
        "GATGraphTransformer": GATGraphTransformer,
    }
    try:
        model_class = models[model_type]
    except KeyError as exc:
        supported = ", ".join(models)
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose from: {supported}"
        ) from exc
    return model_class(hidden_channels=hidden_channels, data=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Graph model for graph classification")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels in the GCN")
    parser.add_argument("--dataset_path", type=str, default="data/", help="Path to the dataset")
    args = parser.parse_args()
    
    data = MoleculeDataset(root=args.dataset_path, filename="bace_processed.csv")

    model = GCN(hidden_channels=args.hidden_channels, data=data)
    print(model)
    
# Example usage:
# python src/model.py --hidden_channels 128 --dataset_path data/