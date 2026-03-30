import torch
import torch.nn as nn
from .gnn_models import GCNModel, GATModel, SAGEModel

class DepressionClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, model_type='gcn'):
        super(DepressionClassifier, self).__init__()
        self.model_type = model_type
        
        if model_type == 'gcn':
            self.gnn = GCNModel(in_channels, hidden_channels, hidden_channels)
        elif model_type == 'gat':
            self.gnn = GATModel(in_channels, hidden_channels, hidden_channels)
        elif model_type == 'sage':
            self.gnn = SAGEModel(in_channels, hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
