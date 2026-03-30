import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool

# GCN模型
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# GAT模型
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# GraphSAGE模型
class SAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x
