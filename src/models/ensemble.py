import torch
import torch.nn as nn
from .base import DepressionClassifier

# 模型集成
class EnsembleClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(EnsembleClassifier, self).__init__()
        # 创建多个不同架构的模型
        self.models = nn.ModuleList([
            DepressionClassifier(in_channels, hidden_channels, num_classes, model_type='gcn'),
            DepressionClassifier(in_channels, hidden_channels, num_classes, model_type='gat'),
            DepressionClassifier(in_channels, hidden_channels, num_classes, model_type='sage')
        ])
    
    def forward(self, x, edge_index, batch):
        # 聚合多个模型的输出
        outputs = []
        for model in self.models:
            outputs.append(model(x, edge_index, batch))
        # 取平均值
        return torch.stack(outputs).mean(dim=0)
