"""
Comprehensive Graph-Based Machine Learning in Python

This script covers various graph-based ML techniques, including:
- Graph construction and visualization
- Graph embeddings (Node2Vec, DeepWalk)
- Graph Neural Networks (GCNs, GATs)
- Community detection algorithms
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. Graph Construction and Visualization
# ----------------------------

# Create a directed graph
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (2, 5), (5, 6)])

# Visualizing the graph
plt.figure(figsize=(6, 4))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=12)
plt.show()

# ----------------------------
# 2. Graph Embeddings (Node2Vec, DeepWalk)
# ----------------------------

from node2vec import Node2Vec

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get vector for a node
node_vector = model.wv[1]

# ----------------------------
# 3. Graph Neural Networks (GCNs, GATs)
# ----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data

# Create a sample graph in PyTorch Geometric format
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
x = torch.rand((4, 5))  # 4 nodes, 5 features per node

data = Data(x=x, edge_index=edge_index)

# Define a GCN model
class GCN(nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model_gcn = GCN(num_features=5)

# Define a GAT model
class GAT(nn.Module):
    def __init__(self, num_features):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=2)
        self.conv2 = GATConv(16 * 2, 2, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model_gat = GAT(num_features=5)

# ----------------------------
# 4. Community Detection Algorithms
# ----------------------------

from networkx.algorithms import community

# Detect communities using the Girvan-Newman method
comp = community.girvan_newman(G)
top_level_communities = next(comp)
print("Detected Communities:", sorted(map(sorted, top_level_communities)))
