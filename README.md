# 基于图神经网络的面部表情特征抑郁症分级方法研究

## 项目简介

本项目实现了一个基于图神经网络（GNN）的抑郁症分级系统，利用面部表情特征（Action Units - AUs）对抑郁症进行5级分类（无抑郁、轻度、中度、中重度、重度）。项目采用多模型集成策略，结合GCN、GAT和GraphSAGE三种图神经网络架构，实现了高精度的抑郁症分级预测。

### 项目创新点
- **多模型集成**：融合三种主流图神经网络模型，提高预测稳定性和准确性
- **面部表情特征**：专注于面部表情动作单元（AUs）作为核心特征，捕捉情绪变化
- **图结构建模**：将面部表情特征构建为图结构，更好地捕捉特征间的关联关系
- **数据增强**：通过滑动窗口和噪声添加技术增加样本多样性
- **全面评估**：提供详细的模型性能分析和可视化结果

### 应用价值
- 为抑郁症的早期筛查和诊断提供辅助工具
- 减少人工诊断的主观性和误差
- 为心理健康领域的研究提供新的技术路径
- 可扩展到其他情绪识别和精神疾病评估场景

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

#### 系统要求
- Python 3.7+ 
- CUDA 10.1+（推荐，用于GPU加速）
- 至少8GB内存
- 10GB可用磁盘空间

#### 安装步骤

1. **创建虚拟环境**（推荐）：
```bash
# 使用conda创建虚拟环境
conda create -n depression-gnn python=3.8
conda activate depression-gnn

# 或使用venv创建虚拟环境
python -m venv venv
# Windows激活
env\Scripts\activate
# Linux/Mac激活
# source venv/bin/activate
```

2. **安装依赖包**：
```bash
pip install -r requirements.txt
```

3. **安装torch-geometric相关依赖**：
```bash
# 根据PyTorch版本安装对应版本的torch-geometric
# 例如，对于PyTorch 1.9.0和CUDA 10.2
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install torch-geometric
```

#### 依赖包说明
- **torch**：PyTorch深度学习框架，提供张量计算和自动微分
- **torch-geometric**：图神经网络库，提供图数据结构和GNN模型
- **pandas**：数据处理和分析库
- **numpy**：数值计算库
- **scikit-learn**：机器学习工具，提供模型评估和传统算法
- **librosa**：音频处理库（可选，用于处理音频特征）
- **matplotlib**：数据可视化库，用于生成结果图表
- **seaborn**：统计数据可视化库，增强图表美观度

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

### 运行示例

#### 基本运行
```bash
# 执行主程序
python scripts/main.py
```

#### 运行参数配置

修改 `config/config.py` 中的配置参数：

```python
# 模型配置
MODEL_CONFIG = {
    'input_channels': 17,        # 输入特征维度（AU特征数量）
    'hidden_channels': 64,       # 隐藏层维度
    'num_classes': 5,            # 分类数
    'model_type': 'ensemble'     # 模型类型：'gcn', 'gat', 'sage', 'ensemble'
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 50,            # 训练轮数
    'batch_size': 8,             # 批次大小
    'learning_rate': 0.001,      # 学习率
    'early_stopping_patience': 10, # 早停耐心值
    'k_folds': 5                 # 交叉验证折数
}
```

#### 运行流程

程序运行时的执行流程：
1. **数据加载**：从配置的路径加载AVEC2017数据集
2. **特征处理**：提取面部表情AU特征，构建图结构
3. **模型训练**：使用5折交叉验证训练模型
4. **模型评估**：计算各项性能指标，与传统模型对比
5. **结果生成**：生成分类报告、混淆矩阵和各种可视化图表
6. **模型保存**：将训练好的模型保存到results目录

#### 预期输出

运行完成后，您将在 `results/` 目录中看到以下文件：
- `models/`：保存的模型文件
- `reports/`：评价报告文件
- `figures/`：可视化图表
  - `confusion_matrix.png`：混淆矩阵
  - `cross_validation_accuracy.png`：交叉验证精度图
  - `roc_curve.png`：ROC曲线
  - `model_comparison.png`：模型对比图
  - `label_distribution.png`：标签分布图
  - `metrics_comparison.png`：指标对比图

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

## 结果解释

### 评价报告解读

运行完成后生成的 `classification_report.txt` 文件包含以下信息：
- **精确率（Precision）**：每个类别的精确率，表示预测为该类别的样本中实际为该类别的比例
- **召回率（Recall）**：每个类别的召回率，表示实际为该类别的样本中被正确预测的比例
- **F1分数（F1 Score）**：精确率和召回率的调和平均，综合反映模型性能
- **支持数（Support）**：每个类别的样本数量

### 可视化结果解读

#### 混淆矩阵
- 对角线元素表示正确分类的样本数
- 非对角线元素表示分类错误的样本数
- 颜色深浅表示样本数量的多少

#### ROC曲线
- 曲线越靠近左上角，模型性能越好
- AUC值越接近1，模型的区分能力越强
- 宏平均ROC曲线反映模型在所有类别上的整体表现

#### 模型对比图
- 展示集成模型与传统机器学习模型的性能对比
- 帮助评估GNN模型的优势

## 实现方法与参数详解

### 1. 特征提取方法

#### 面部表情特征（AU）提取
- **特征来源**：从AVEC2017数据集中的`[Participant_ID]_CLNF_AUs.txt`文件提取
- **特征数量**：17个面部表情动作单元（AUs）
- **处理方法**：
  - 滑动窗口：使用大小为30的窗口，步长为10
  - 数据增强：添加高斯噪声，增强模型鲁棒性
  - 标准化：对特征进行Z-score标准化

#### 图结构构建
- **节点**：每个节点代表一个时间点的面部表情特征
- **边**：基于时间序列构建边，相邻时间点之间建立连接
- **边权重**：根据时间间隔计算权重，间隔越小权重越大

### 2. 模型架构与参数

#### 基础GNN模型

##### GCN模型（Graph Convolutional Network）
- **参数**：
  - `input_channels`：输入特征维度，默认17
  - `hidden_channels`：隐藏层维度，默认64
  - `num_classes`：分类数，默认5
  - `dropout`：dropout概率，默认0.5
- **结构**：
  - 两层图卷积层
  - 批归一化
  - ReLU激活函数
  - Dropout层

##### GAT模型（Graph Attention Network）
- **参数**：
  - 同GCN模型参数
  - `heads`：注意力头数，默认4
- **结构**：
  - 两层图注意力层
  - 多头注意力机制
  - 批归一化
  - ReLU激活函数
  - Dropout层

##### SAGE模型（GraphSAGE）
- **参数**：
  - 同GCN模型参数
  - `aggregator_type`：聚合器类型，默认"mean"
- **结构**：
  - 两层GraphSAGE层
  - 均值聚合器
  - 批归一化
  - ReLU激活函数
  - Dropout层

#### 集成模型
- **集成策略**：平均多个模型的输出概率
- **成员模型**：GCN、GAT、SAGE
- **优势**：减少模型方差，提高预测稳定性

### 3. 训练方法与参数

#### 训练配置
- **`num_epochs`**：训练轮数，默认50
- **`batch_size`**：批次大小，默认8
- **`learning_rate`**：学习率，默认0.001
- **`early_stopping_patience`**：早停耐心值，默认10
- **`k_folds`**：交叉验证折数，默认5

#### 优化器与损失函数
- **优化器**：Adam优化器
- **损失函数**：交叉熵损失
- **学习率调度**：固定学习率

#### 评估方法
- **5折交叉验证**：确保模型泛化能力
- **评价指标**：准确率、精确率、召回率、F1分数、AUC值
- **基准对比**：与逻辑回归、随机森林、SVM等传统模型对比

### 4. 数据处理参数

#### 数据路径配置
- **`train_labels_path`**：训练集标签文件路径
- **`dev_labels_path`**：验证集标签文件路径
- **`test_labels_path`**：测试集标签文件路径
- **`data_dir`**：数据根目录

#### 特征配置
- **`window_size`**：滑动窗口大小，默认30
- **`step_size`**：滑动窗口步长，默认10
- **`augmentation`**：数据增强开关，默认True
- **`noise_std`**：噪声标准差，默认0.01

#### 抑郁等级划分
- **0级**：无抑郁（BDI-II < 14）
- **1级**：轻度抑郁（14 ≤ BDI-II < 20）
- **2级**：中度抑郁（20 ≤ BDI-II < 28）
- **3级**：中重度抑郁（28 ≤ BDI-II < 35）
- **4级**：重度抑郁（BDI-II ≥ 35）

### 5. 可视化方法

#### 混淆矩阵
- **方法**：`plot_confusion_matrix()`
- **参数**：
  - `y_true`：真实标签
  - `y_pred`：预测标签
  - `classes`：类别名称
  - `title`：图表标题
- **输出**：混淆矩阵热力图

#### ROC曲线
- **方法**：`plot_roc_curve()`
- **参数**：
  - `y_true`：真实标签
  - `y_score`：预测概率
  - `classes`：类别名称
- **输出**：各类别ROC曲线和AUC值

#### 模型对比
- **方法**：`plot_model_comparison()`
- **参数**：
  - `model_names`：模型名称列表
  - `metrics`：评价指标数据
- **输出**：模型性能对比柱状图

### 6. 主程序流程

1. **数据加载**：`load_data()`函数，加载AVEC2017数据集
2. **特征处理**：`process_features()`函数，提取和处理AU特征
3. **图构建**：`build_graph()`函数，构建时间序列图结构
4. **模型初始化**：根据配置创建指定类型的模型
5. **交叉验证**：`cross_validate()`函数，执行5折交叉验证
6. **模型训练**：`train()`函数，训练模型并记录性能
7. **模型评估**：`test()`函数，评估模型性能
8. **结果生成**：`generate_results()`函数，生成评价报告和可视化图表
9. **模型保存**：`save_model()`函数，保存训练好的模型

### 7. 关键函数详解

#### `create_dataset()`
- **功能**：将处理后的数据转换为PyTorch Geometric数据集
- **参数**：
  - `features`：特征矩阵
  - `labels`：标签向量
  - `edge_index`：边索引
  - `edge_weight`：边权重
- **返回值**：PyTorch Geometric Data对象

#### `train()`
- **功能**：训练模型
- **参数**：
  - `model`：模型对象
  - `train_loader`：训练数据加载器
  - `optimizer`：优化器
  - `criterion`：损失函数
  - `device`：设备（CPU/GPU）
- **返回值**：训练损失和准确率

#### `test()`
- **功能**：测试模型
- **参数**：
  - `model`：模型对象
  - `test_loader`：测试数据加载器
  - `criterion`：损失函数
  - `device`：设备（CPU/GPU）
- **返回值**：测试损失、准确率、预测结果和概率

#### `ensemble_predict()`
- **功能**：集成模型预测
- **参数**：
  - `models`：模型列表
  - `data`：输入数据
  - `device`：设备（CPU/GPU）
- **返回值**：集成预测结果和概率

## 更新日志

### v1.0.0 (2026-04-03)
- 项目初始化
- 实现基于GCN、GAT、GraphSAGE的集成模型
- 支持AVEC2017数据集处理
- 实现5折交叉验证
- 生成多种可视化结果
- 与传统机器学习模型对比

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，欢迎提出Issue或Pull Request。

## 致谢

- 感谢AVEC2017数据集的提供者
- 感谢PyTorch和PyTorch Geometric团队的开源贡献
- 感谢所有为本项目提供支持和建议的同事和朋友
