# 数据配置

import os

# 数据路径
DATA_CONFIG = {
    'data_dir': os.path.join(os.path.dirname(__file__), '..', 'data', 'AVEC2017'),
    'train_labels': 'train_split_Depression_AVEC2017.csv',
    'dev_labels': 'dev_split_Depression_AVEC2017.csv',
    'test_labels': 'test_split_Depression_AVEC2017.csv'
}

# 特征提取配置
FEATURE_CONFIG = {
    'window_size': 5,  # 滑动窗口大小
    'stride': 2,  # 滑动步长
    'augment': True,  # 是否进行数据增强
    'max_samples_per_subject': 10  # 每个受试者最大样本数
}

# 特征类型
FEATURE_TYPES = {
    'au': 'CLNF_AUs.txt',
    'hog': 'CLNF_hog',  # 支持.txt和.bin格式
    'covarep': 'COVAREP.csv',
    'formant': 'FORMANT.csv',
    'audio': 'AUDIO.wav'
}

# 抑郁等级划分
DEPRESSION_LEVELS = {
    'no_depression': 0,  # 0-4分
    'mild': 1,  # 5-9分
    'moderate': 2,  # 10-14分
    'moderate_severe': 3,  # 15-19分
    'severe': 4  # 20-24分
}
