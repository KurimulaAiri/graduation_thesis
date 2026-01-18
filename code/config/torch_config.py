import torch

# 数据预处理配置
INVALID_SESSIONS = [342, 394, 398, 460]  # 无效会话（需排除）
CONFIDENCE_THRESHOLD = 0.8  # 有效帧筛选阈值（CLNF跟踪置信度）
DEPRESSION_LEVELS = {0: (0, 4), 1: (5, 9), 2: (10, 14), 3: (15, 27)}  # 4级分级标准
SELECTED_AUs = ["AU01_r", "AU04_r", "AU12_r", "AU15_r", "AU01_c", "AU04_c", "AU12_c", "AU15_c"]  # 抑郁敏感AU

# 图结构配置
NODE_TYPE = "AU"  # 节点类型："AU"（推荐）或 "KEYPOINT"（68关键点）
EDGE_WEIGHT = {  # AU节点加权边配置（临床相关性）
    ("AU01", "AU04"): 0.8, ("AU04", "AU15"): 0.9, ("AU12", "AU01"): 0.3,
    ("AU12", "AU15"): 0.2, ("AU01", "AU15"): 0.7, ("AU04", "AU12"): 0.4
}

# 模型配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 2  # 每个节点特征维度（AU：强度_r+出现_c；关键点：x+y）
HIDDEN_DIM = 128
NUM_CLASSES = 4  # 4级分级
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15  # 早停耐心值