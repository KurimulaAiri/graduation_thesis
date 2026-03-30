# 模型和训练配置

# 模型配置
MODEL_CONFIG = {
    'in_channels': 0,  # 将在运行时根据数据动态设置
    'hidden_channels': 64,
    'num_classes': 5,  # 5个抑郁等级
    'model_type': 'ensemble'  # 可选: 'gcn', 'gat', 'sage', 'ensemble'
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 50,
    'batch_size': 8,
    'learning_rate': 0.001,
    'weight_decay': 0.001,
    'patience': 10,  # 早停耐心
    'k_folds': 5,  # 交叉验证折数
    'random_state': 42
}

# 设备配置
DEVICE = 'cpu'  # 可以设置为 'cuda' 如果有GPU

# 结果保存配置
RESULTS_CONFIG = {
    'save_dir': 'results',
    'save_model': True,
    'save_plots': True,
    'save_report': True
}
