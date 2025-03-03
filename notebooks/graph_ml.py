"""
Graph-Based Machine Learning in Python

This script covers various graph-based ML techniques, including:
- Graph construction
- Graph embeddings (Node2Vec, DeepWalk)
- Graph Neural Networks (GNNs)
"""

import networkx as nx
import numpy as np

# ----------------------------
# 1. Graph Construction
# ----------------------------

G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

# ----------------------------
# 2. Graph Embeddings (Node2Vec)
# ----------------------------

from node2vec import Node2Vec

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# ----------------------------
# 3. Graph Neural Networks (GNNs)
# ----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(num_features=10)
