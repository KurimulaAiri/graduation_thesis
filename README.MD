## 项目结构（暂定）

```PLAINTEXT
depression_gnn/
├── 01_data/                  # 数据相关（原始数据、预处理结果、图数据）
│   ├── raw/                  # 原始数据（未处理的图像/视频、标签文件）
│   │   ├── images/           # 面部图像（如抑郁患者/正常对照组的面部照片）
│   │   ├── videos/           # 面部视频（动态表情数据，可选）
│   │   └── labels.csv        # 标签文件（图像路径+抑郁分级标签+临床信息）
│   ├── processed/            # 预处理后的数据（中间结果，可复用）
│   │   ├── landmarks/        # 提取的面部关键点（npy格式，对应每张图像）
│   │   └── metadata.csv      # 预处理元数据（如是否成功提取关键点、异常标注）
│   └── graph_data/           # 构建的图结构数据（PyTorch Geometric格式）
│       ├── train/            # 训练集图数据
│       ├── val/              # 验证集图数据
│       └── test/             # 测试集图数据
│
├── 02_code/                  # 核心代码模块（按功能拆分，避免单文件冗长）
│   ├── __init__.py           # 声明为Python包，支持跨模块导入
│   ├── config.py             # 配置参数（路径、超参数、常量等，集中管理）
│   ├── data_utils/           # 数据预处理工具
│   │   ├── face_extractor.py # 面部关键点提取（封装dlib/opencv逻辑）
│   │   ├── graph_builder.py  # 图结构构建（节点/边定义、特征处理）
│   │   └── dataset.py        # 数据集类（继承PyTorch Geometric的Dataset）
│   │
│   ├── models/               # 模型定义
│   │   ├── base_gnn.py       # 基础GNN模型（如GCN、GAT的简化实现）
│   │   ├── depression_gnn.py # 项目核心模型（如带注意力的GNN、时空融合模型）
│   │   └── layers.py         # 自定义层（如特殊的图卷积、注意力层）
│   │
│   ├── train_eval/           # 训练与评估逻辑
│   │   ├── trainer.py        # 训练循环（含损失函数、优化器、早停等）
│   │   ├── evaluator.py      # 评估函数（计算指标、混淆矩阵、可视化）
│   │   └── utils.py          # 辅助函数（如学习率调度、模型保存/加载）
│   │
│   └── visualization/        # 可视化工具（辅助分析）
│       ├── graph_viz.py      # 图结构可视化（如节点注意力权重热力图）
│       └── result_viz.py     # 结果可视化（如训练曲线、混淆矩阵图）
│
├── 03_scripts/               # 可直接运行的脚本（调用02_code中的模块）
│   ├── preprocess.py         # 数据预处理总脚本（从原始图像→图数据）
│   ├── train.py              # 模型训练脚本（指定参数、调用trainer）
│   ├── evaluate.py           # 模型评估脚本（在测试集上运行并输出报告）
│   └── demo.py               # 演示脚本（单样本预测，用于展示效果）
│
├── 04_results/               # 实验结果（确保可复现）
│   ├── models/               # 保存的模型权重（按实验ID/日期命名）
│   ├── metrics/              # 评估指标（CSV格式，含各实验的准确率、F1等）
│   ├── figures/              # 可视化结果（训练曲线、混淆矩阵图、注意力热力图）
│   └── logs/                 # 训练日志（记录超参数、每轮损失和指标）
│
├── 05_notebooks/             # 实验性笔记本（用于快速验证想法）
│   ├── 01_eda.ipynb          # 探索性数据分析（数据分布、关键点可视化）
│   ├── 02_graph_demo.ipynb   # 图结构构建示例（展示节点/边设计合理性）
│   └── 03_model_ablation.ipynb # 消融实验（验证各模块作用，如注意力机制的影响）
│
├── requirements.txt          # 依赖库列表（精确到版本，确保环境一致）
├── README.md                 # 项目说明（环境配置、运行步骤、文件结构解释）
└── LICENSE                   # 许可证（若开源，需声明，如MIT）
```