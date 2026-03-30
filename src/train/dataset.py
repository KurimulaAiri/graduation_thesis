import os
import numpy as np
import torch
from torch_geometric.data import Data

def create_dataset(subjects):
    """
    创建PyTorch Geometric数据集
    """
    data_list = []
    valid_label_count = 0
    valid_graph_count = 0
    
    for i, subject in enumerate(subjects):
        # 每处理1000个样本打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"处理中：{i + 1}/{len(subjects)} 个样本")
        
        # 跳过没有有效标签的样本
        if subject['label'] == -1:
            continue
        valid_label_count += 1
        
        # 限制节点数量，减少内存使用
        node_features = subject['node_features'][:50]  # 只使用前50个节点
        edges = subject['edges']
        
        # 检查节点特征
        if len(node_features) == 0:
            continue
        if len(node_features[0]) == 0:
            continue
        
        # 处理边
        if edges is None:
            print(f"受试者 {subject['id']} 的边为None")
            continue
        
        # 转换为列表格式
        if isinstance(edges, np.ndarray):
            edges = edges.tolist()
        
        # 检查边的格式
        if not isinstance(edges, list):
            print(f"受试者 {subject['id']} 的边格式不正确: {type(edges)}")
            continue
        
        # 过滤边，只保留有效的边
        valid_nodes = set(range(len(node_features)))
        valid_edges = []
        for edge in edges:
            if len(edge) == 2 and edge[0] in valid_nodes and edge[1] in valid_nodes:
                valid_edges.append(edge)
        
        if len(valid_edges) == 0:
            print(f"受试者 {subject['id']} 没有有效的边")
            print(f"原始边数量: {len(edges)}, 节点数量: {len(node_features)}")
            if len(edges) > 0:
                print(f"第一条边: {edges[0]}, 最后一条边: {edges[-1]}")
            continue
        valid_graph_count += 1
        print(f"受试者 {subject['id']} 创建了有效的图结构，边数量: {len(valid_edges)}")
        
        # 优化张量创建方式
        try:
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            edge_index = torch.tensor(np.array(valid_edges), dtype=torch.long).t().contiguous()
            # 使用真实标签
            y = torch.tensor([subject['label']], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        except Exception as e:
            print(f"创建数据时出错：{str(e)}")
            continue
    
    print(f"有效标签数量：{valid_label_count}")
    print(f"有效图结构数量：{valid_graph_count}")
    return data_list

def extract_flatten_features(dataset):
    """
    提取扁平化特征，用于传统机器学习模型
    """
    X = []
    y = []
    for data in dataset:
        # 提取节点特征的均值作为扁平化特征
        feature = torch.mean(data.x, dim=0).numpy()
        X.append(feature)
        y.append(data.y.item())
    return np.array(X), np.array(y)
