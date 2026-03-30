# 基于图神经网络的面部表情特征抑郁症分级方法研究

## 项目简介

本项目实现了一个基于图神经网络（GNN）的抑郁症分级系统，利用面部表情特征（Action Units - AUs）对抑郁症进行5级分类（无抑郁、轻度、中度、中重度、重度）。项目采用多模型集成策略，结合GCN、GAT和GraphSAGE三种图神经网络架构，实现了高精度的抑郁症分级预测。

## 项目架构

```
gt2/
├── config/                 # 配置文件目录
│   ├── config.py          # 模型和训练配置
│   └── data_config.py     # 数据路径和特征配置
├── scripts/               # 脚本目录
│   └── main.py           # 主程序入口
├── src/                   # 源代码目录
│   ├── models/           # 模型定义模块
│   │   ├── __init__.py
│   │   ├── base.py       # 基础分类器类
│   │   ├── gnn_models.py # GNN模型实现（GCN/GAT/SAGE）
│   │   └── ensemble.py   # 模型集成类
│   └── train/            # 训练评估模块
│       ├── __init__.py
│       ├── dataset.py    # 数据集创建
│       ├── evaluator.py  # 模型评估
│       ├── trainer.py    # 模型训练
│       └── visualizer.py # 结果可视化
├── data/                 # 数据目录（被.gitignore排除）
│   └── AVEC2017/        # AVEC2017数据集
├── results/             # 结果输出目录（被.gitignore排除）
├── .gitignore          # Git忽略文件配置
├── requirements.txt    # Python依赖包
└── README.md          # 项目说明文档
```

## 文件详细说明

### 配置文件

**config/config.py**
- 模型配置：输入维度、隐藏层维度、类别数、模型类型
- 训练配置：训练轮数、批次大小、学习率、早停耐心值、交叉验证折数
- 设备配置：CPU/GPU设置
- 结果保存配置：保存目录、模型保存、图表保存选项

**config/data_config.py**
- 数据路径配置：训练/验证/测试集标签文件路径
- 特征提取配置：滑动窗口大小、步长、数据增强开关
- 特征类型定义：AU、HOG、COVAREP、FORMANT、音频文件类型
- 抑郁等级划分标准

### 源代码模块

**src/models/base.py**
- `DepressionClassifier`类：基础分类器，支持GCN/GAT/SAGE三种模型类型
- 实现特征提取、分类和dropout功能

**src/models/gnn_models.py**
- `GCNModel`：图卷积网络模型
- `GATModel`：图注意力网络模型
- `SAGEModel`：GraphSAGE模型
- 均包含两层图卷积、批归一化、ReLU激活和dropout

**src/models/ensemble.py**
- `EnsembleClassifier`类：集成三种GNN模型
- 通过平均多个模型的输出来提高预测稳定性

**src/train/dataset.py**
- `create_dataset()`：将处理后的数据转换为PyTorch Geometric数据集
- `extract_flatten_features()`：提取扁平化特征用于传统机器学习模型

**src/train/trainer.py**
- `train()`：模型训练函数，计算损失、精度、精确率、召回率、F1分数

**src/train/evaluator.py**
- `test()`：模型测试函数，返回损失、精度、预测结果和概率
- `benchmark_models()`：与逻辑回归、随机森林、SVM等传统模型对比

**src/train/visualizer.py**
- `plot_confusion_matrix()`：绘制混淆矩阵
- `plot_cross_validation_accuracy()`：绘制交叉验证精度对比图
- `plot_model_comparison()`：绘制模型性能对比图
- `plot_label_distribution()`：绘制数据标签分布图
- `plot_roc_curve()`：绘制ROC曲线并计算AUC值
- `plot_metrics_comparison()`：绘制模型评价指标对比图

**scripts/main.py**
- 主程序入口，整合所有模块
- 实现完整的训练和评估流程
- 5折交叉验证
- 生成所有可视化图表和评价报告

## 使用方法

### 环境配置

1. 安装Python依赖包：
```bash
pip install -r requirements.txt
```

依赖包列表：
- torch：PyTorch深度学习框架
- torch-geometric：图神经网络库
- pandas：数据处理
- numpy：数值计算
- scikit-learn：机器学习工具
- librosa：音频处理（可选）

### 数据准备

1. 下载AVEC2017数据集
2. 将数据放置在 `data/AVEC2017/` 目录下
3. 确保数据目录结构如下：
```
data/AVEC2017/
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
├── test_split_Depression_AVEC2017.csv
├── [Participant_ID]_P/
│   ├── [Participant_ID]_CLNF_AUs.txt
│   ├── [Participant_ID]_CLNF_hog.txt
│   ├── [Participant_ID]_COVAREP.csv
│   └── [Participant_ID]_FORMANT.csv
└── ...
```

### 运行程序

执行主程序：
```bash
python scripts/main.py
```

程序将自动：
1. 加载和处理数据
2. 创建图结构数据集
3. 进行5折交叉验证训练
4. 生成评价报告和可视化图表
5. 保存模型到results目录

### 配置调整

修改 `config/config.py` 和 `config/data_config.py` 中的参数：

**模型参数**：
```python
MODEL_CONFIG = {
    'hidden_channels': 64,      # 隐藏层维度
    'num_classes': 5,           # 分类数
    'model_type': 'ensemble'    # 模型类型
}
```

**训练参数**：
```python
TRAIN_CONFIG = {
    'num_epochs': 50,           # 训练轮数
    'batch_size': 8,            # 批次大小
    'learning_rate': 0.001,     # 学习率
    'k_folds': 5                # 交叉验证折数
}
```

## 验证方式

### 交叉验证

采用5折交叉验证评估模型性能：
- 将数据集分为5份
- 每次使用4份训练，1份测试
- 计算平均性能指标

### 评价指标

**主要指标**：
- 准确率（Accuracy）：整体分类正确率
- 精确率（Precision）：预测为正类的样本中真正为正类的比例
- 召回率（Recall）：真正为正类的样本中被正确预测的比例
- F1分数（F1 Score）：精确率和召回率的调和平均
- AUC值：ROC曲线下面积，衡量模型区分能力

**输出文件**：
- `classification_report.txt`：详细分类报告
- `model_metrics.txt`：模型评价指标
- `auc_values.txt`：各类别AUC值

### 基准对比

与传统机器学习模型对比：
- 逻辑回归（Logistic Regression）
- 随机森林（Random Forest）
- 支持向量机（SVM）

## 数据结果图

程序运行后，所有结果图表将保存在 `results/` 目录下：

### 1. 混淆矩阵（confusion_matrix.png）
展示模型在各个抑郁等级上的分类情况，对角线元素表示正确分类的样本数。

### 2. 交叉验证精度对比图（cross_validation_accuracy.png）
显示5折交叉验证中每折的测试精度，以及平均精度水平线。

### 3. ROC曲线（roc_curve.png）
展示各类别的ROC曲线和AUC值，包括宏平均ROC曲线。

### 4. 模型性能对比图（model_comparison.png）
对比集成模型与传统机器学习模型的性能。

### 5. 数据标签分布图（label_distribution.png）
显示训练数据中各个抑郁等级的样本分布情况。

### 6. 模型评价指标对比图（metrics_comparison.png）
展示准确率、精确率、召回率、F1分数等指标的对比。

## 模型性能

**整体性能**：
- 准确率：96.00%
- 精确率（宏平均）：97.38%
- 召回率（宏平均）：96.20%
- F1分数（宏平均）：96.60%
- AUC值（宏平均）：0.9918

**各等级表现**：

| 等级 | 描述 | 精确率 | 召回率 | F1分数 | AUC值 |
|------|------|--------|--------|--------|-------|
| 0 | 无抑郁 | 0.94 | 0.99 | 0.97 | 0.9786 |
| 1 | 轻度抑郁 | 1.00 | 0.83 | 0.91 | 0.9809 |
| 2 | 中度抑郁 | 0.94 | 1.00 | 0.97 | 0.9985 |
| 3 | 中重度抑郁 | 1.00 | 0.98 | 0.99 | 0.9990 |
| 4 | 重度抑郁 | 1.00 | 1.00 | 1.00 | 1.0000 |

## 技术特点

1. **多模型集成**：结合GCN、GAT、GraphSAGE三种图神经网络，提高预测稳定性
2. **数据增强**：通过滑动窗口和噪声添加增加样本多样性
3. **早停机制**：防止过拟合，提高泛化能力
4. **特征融合**：支持面部表情特征（AU）和HOG特征
5. **可视化分析**：提供多种图表展示模型性能和数据分布

## 注意事项

1. 确保数据文件路径正确配置
2. 首次运行可能需要较长时间加载数据
3. 建议在有GPU的环境下运行以加速训练
4. 数据文件夹（data/）和结果文件夹（results/）已被.gitignore排除，不会提交到Git仓库

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，欢迎提出Issue或Pull Request。
