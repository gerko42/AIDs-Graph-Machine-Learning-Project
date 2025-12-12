import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import GraphConv, GATConv

class GCNGraphClassifier(nn.Module):
    """Graph Convolutional Network for graph classification"""
    
    def __init__(self, in_features, hidden_dim=64, num_classes=2, num_layers=3):
        super(GCNGraphClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification layers
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        """Forward pass
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
        """
        # Graph convolutions
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification head
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

class GATGraphClassifier(nn.Module):
    """Graph Attention Network for graph classification"""
    
    def __init__(self, in_features, hidden_dim=64, num_classes=2, num_layers=3, num_heads=4):
        super(GATGraphClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GATConv(in_features, hidden_dim // num_heads, heads=num_heads, dropout=0.5))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.5))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification layers
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        """Forward pass"""
        # Graph attention layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification head
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x
