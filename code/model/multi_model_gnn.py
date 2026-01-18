import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import warnings
import gc
import traceback

from code.utils import Read_HOG_files
import code.config as cfg

warnings.filterwarnings("ignore")


# ===================== 1. 配置 + GPU设置 ======================
class Config:
    DATA_ROOT = r"D:\PRJ\pythonPRJ\graduation_thesis\data\AVEC2017"
    SPLIT_ROOT = r"D:\PRJ\pythonPRJ\graduation_thesis\data\AVEC2017"
    # 仅作为参考，不再硬编码计算总维度
    HOG_DIM_REF = 4464
    COVAREP_DIM_REF = 63
    HIDDEN_DIM = 128
    NUM_CLASSES = 4
    BATCH_SIZE = 2
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = True if torch.cuda.is_available() else False

    # 内存优化参数
    HOG_DOWNSAMPLE_RATE = 10
    USE_FLOAT16 = True
    MAX_HOG_MEMORY_MB = 100


config = Config()
print(f"当前设备: {config.DEVICE}")


# ===================== 2. 特征读取函数（增加维度打印） ======================
def read_hog_file(hog_path):
    hog_dir = os.path.dirname(hog_path)
    subj_id = os.path.basename(hog_path).split('_')[0]
    user_name = f"{subj_id}_CLNF_hog.bin"

    hog_dict = Read_HOG_files(
        users=[user_name],
        hog_data_dir=hog_dir,
        downsample_rate=config.HOG_DOWNSAMPLE_RATE,
        use_float16=config.USE_FLOAT16,
        max_memory_mb=config.MAX_HOG_MEMORY_MB
    )

    if user_name not in hog_dict or hog_dict[user_name].size == 0:
        raise FileNotFoundError(f"HOG文件 {hog_path} 读取失败或为空")

    hog_data = hog_dict[user_name]
    if config.USE_FLOAT16:
        hog_data = hog_data.astype(np.float32)

    # 打印实际HOG维度（调试关键）
    print(f"📏 {user_name} 实际HOG维度：{hog_data.shape[1]}（参考值：{config.HOG_DIM_REF}）")

    # 强制对齐到参考维度（补0/截断）
    if hog_data.shape[1] != config.HOG_DIM_REF:
        if hog_data.shape[1] > config.HOG_DIM_REF:
            hog_data = hog_data[:, :config.HOG_DIM_REF]
            print(f"⚠️  {user_name} HOG维度截断至：{config.HOG_DIM_REF}")
        else:
            pad_width = ((0, 0), (0, config.HOG_DIM_REF - hog_data.shape[1]))
            hog_data = np.pad(hog_data, pad_width, mode='constant')
            print(f"⚠️  {user_name} HOG维度补0至：{config.HOG_DIM_REF}")

    print(f"最终HOG特征：{user_name} → 形状：{hog_data.shape}")
    return hog_data


def read_covarep_file(covarep_path):
    df = pd.read_csv(covarep_path, header=None)
    covarep_data = df.values.astype(np.float32)
    covarep_data = np.nan_to_num(covarep_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 下采样
    if len(covarep_data) > 0:
        covarep_data = covarep_data[::config.HOG_DOWNSAMPLE_RATE]

    # 打印实际COVAREP维度（调试关键）
    print(
        f"📏 {os.path.basename(covarep_path)} 实际COVAREP维度：{covarep_data.shape[1]}（参考值：{config.COVAREP_DIM_REF}）")

    # 强制对齐到参考维度
    if covarep_data.shape[1] != config.COVAREP_DIM_REF:
        if covarep_data.shape[1] > config.COVAREP_DIM_REF:
            covarep_data = covarep_data[:, :config.COVAREP_DIM_REF]
            print(f"⚠️  COVAREP维度截断至：{config.COVAREP_DIM_REF}")
        else:
            pad_width = ((0, 0), (0, config.COVAREP_DIM_REF - covarep_data.shape[1]))
            covarep_data = np.pad(covarep_data, pad_width, mode='constant')
            print(f"⚠️  COVAREP维度补0至：{config.COVAREP_DIM_REF}")

    return covarep_data


def align_frames(visual_data, audio_data):
    min_frames = min(visual_data.shape[0], audio_data.shape[0])
    return visual_data[:min_frames], audio_data[:min_frames]


def phq8_to_level(phq8_score):
    if phq8_score <= 4:
        return 0
    elif 5 <= phq8_score <= 9:
        return 1
    elif 10 <= phq8_score <= 14:
        return 2
    else:
        return 3


# ===================== 3. 数据集类（增加维度记录） ======================
class DAICWOZDataset(Dataset):
    def __init__(self, subject_ids, split_type="train"):
        super().__init__()
        self.subject_ids = subject_ids
        self.split_type = split_type
        self.scaler = StandardScaler()
        self.data_list = self._build_dataset()
        # 记录实际的输入特征维度（所有样本统一）
        self.input_dim = None
        if len(self.data_list) > 0:
            self.input_dim = self.data_list[0].x.shape[1]
            print(f"\n📌 数据集实际输入维度：{self.input_dim}")

    def _build_dataset(self):
        data_list = []
        print(f"\n处理{self.split_type}集，共{len(self.subject_ids)}个被试")

        # 加载split文件
        split_file = os.path.join(config.SPLIT_ROOT, f"{self.split_type}_split_Depression_AVEC2017.csv")
        if self.split_type == "test":
            split_file = os.path.join(config.SPLIT_ROOT, "full_test_split.csv")

        try:
            split_df = pd.read_csv(split_file)
            print(f"\n📌 {self.split_type}集split文件列名：{split_df.columns.tolist()}")
        except Exception as e:
            print(f"❌ 加载split文件失败：{split_file} - {str(e)}")
            return data_list

        for idx, subj in enumerate(self.subject_ids):
            subj_num = subj.split('_')[0]
            hog_path = os.path.join(config.DATA_ROOT, subj, f"{subj_num}_CLNF_hog.bin")
            covarep_path = os.path.join(config.DATA_ROOT, subj, f"{subj_num}_COVAREP.csv")

            if not os.path.exists(hog_path) or not os.path.exists(covarep_path):
                print(f"⚠️  跳过{subj}：文件缺失")
                continue

            try:
                # 读取特征
                hog_data = read_hog_file(hog_path)
                covarep_data = read_covarep_file(covarep_path)
                hog_aligned, covarep_aligned = align_frames(hog_data, covarep_data)

                if hog_aligned.shape[0] < 10:
                    print(f"⚠️  跳过{subj}：有效帧数不足")
                    continue

                # 拼接特征
                fusion_features = np.hstack([hog_aligned, covarep_aligned])
                # 校验总维度
                total_dim = fusion_features.shape[1]
                expected_dim = config.HOG_DIM_REF + config.COVAREP_DIM_REF
                print(f"📏 {subj} 拼接后总维度：{total_dim}（期望：{expected_dim}）")

                # 标准化
                fusion_features = self.scaler.fit_transform(fusion_features)
                num_frames = fusion_features.shape[0]

                # 读取标签（修复后的逻辑）
                phq8_col = None
                possible_phq8_cols = ['PHQ8_Score', 'PHQ8', 'phq8_score', 'PHQ-8', 'PHQ_8', 'phq8']
                for col in possible_phq8_cols:
                    if col in split_df.columns:
                        phq8_col = col
                        break

                if phq8_col is None:
                    print(f"❌ 跳过{subj}：未找到PHQ8列")
                    continue

                id_col = None
                possible_id_cols = ['Participant_ID', 'participant_ID', 'ParticipantId', 'id', 'PID']
                for col in possible_id_cols:
                    if col in split_df.columns:
                        id_col = col
                        break

                if id_col is None:
                    print(f"❌ 跳过{subj}：未找到ID列")
                    continue

                subj_num_int = int(subj_num)
                split_df[id_col] = pd.to_numeric(split_df[id_col], errors='coerce')
                matched_rows = split_df[split_df[id_col] == subj_num_int]

                if len(matched_rows) == 0:
                    print(f"❌ 跳过{subj}：ID未找到")
                    continue

                phq8_score = matched_rows[phq8_col].values[0]
                if pd.isna(phq8_score):
                    print(f"❌ 跳过{subj}：PHQ8分数为空")
                    continue
                phq8_score = float(phq8_score)

                # 转换分级
                label = phq8_to_level(phq8_score)

                # 构建图数据
                x = torch.tensor(fusion_features, dtype=torch.float32)
                y = torch.tensor([label], dtype=torch.long)

                edge_index = []
                for t in range(num_frames - 1):
                    edge_index.append([t, t + 1])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                if num_frames > 0 and edge_index.size(1) > 0:
                    data_list.append(Data(x=x, edge_index=edge_index, y=y, subject_id=subj))

                # 释放内存
                del hog_data, covarep_data, hog_aligned, covarep_aligned, fusion_features
                gc.collect()

                print(f"✅  处理完成{subj} → 帧数：{num_frames} → PHQ8：{phq8_score} → 分级：{label}")

            except MemoryError:
                print(f"❌  内存不足：{subj}")
                gc.collect()
                continue
            except Exception as e:
                print(f"❌  处理失败：{subj} - {str(e)}")
                traceback.print_exc()
                gc.collect()
                continue

        gc.collect()
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ===================== 4. 加载被试列表 ======================
def load_split_subjects():
    train_df = pd.read_csv(os.path.join(config.SPLIT_ROOT, "train_split_Depression_AVEC2017.csv"))
    dev_df = pd.read_csv(os.path.join(config.SPLIT_ROOT, "dev_split_Depression_AVEC2017.csv"))
    test_df = pd.read_csv(os.path.join(config.SPLIT_ROOT, "test_split_Depression_AVEC2017.csv"))

    train_subjects = [f"{pid}_P" for pid in train_df["Participant_ID"].tolist()]
    dev_subjects = [f"{pid}_P" for pid in dev_df["Participant_ID"].tolist()]
    test_subjects = [f"{pid}_P" for pid in test_df["participant_ID"].tolist()]

    invalid_pids = [342, 394, 398, 460]
    train_subjects = [s for s in train_subjects if int(s.split('_')[0]) not in invalid_pids]
    dev_subjects = [s for s in dev_subjects if int(s.split('_')[0]) not in invalid_pids]
    test_subjects = [s for s in test_subjects if int(s.split('_')[0]) not in invalid_pids]

    return train_subjects, dev_subjects, test_subjects


# ===================== 5. 模型定义（修复维度的注意力GAT） ======================
class AttentionGAT(torch.nn.Module):
    def __init__(self, input_dim, hog_dim_ref=4464, covarep_dim_ref=63, hidden_dim=128, num_classes=4):
        super().__init__()
        self.hog_dim_ref = hog_dim_ref  # HOG参考维度
        self.covarep_dim_ref = covarep_dim_ref  # COVAREP参考维度

        # 1. 模态注意力层（输出2个权重，对应HOG/COVAREP）
        self.attn_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2),  # 输入是总维度，输出2个权重
            torch.nn.Softmax(dim=1)  # 权重归一化
        )

        # 2. GAT层（保持正则化强度）
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=True, dropout=0.4)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False, dropout=0.4)

        # 3. 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # ===== 核心修复1：动态拆分+维度校验 =====
        # 步骤1：拆分HOG和COVAREP（按参考维度）
        hog_feat = x[:, :self.hog_dim_ref]
        covarep_feat = x[:, self.hog_dim_ref:]

        # 步骤2：打印维度（调试关键）
        if batch[0] == 0:  # 只打印第一个batch的维度，避免刷屏
            print(f"\n📏 模型内维度校验：")
            print(f"   总输入维度：{x.shape[1]}")
            print(f"   HOG拆分维度：{hog_feat.shape[1]}（参考：{self.hog_dim_ref}）")
            print(f"   COVAREP拆分维度：{covarep_feat.shape[1]}（参考：{self.covarep_dim_ref}）")

        # 步骤3：强制对齐COVAREP维度（补0/截断）
        if covarep_feat.shape[1] != self.covarep_dim_ref:
            if covarep_feat.shape[1] > self.covarep_dim_ref:
                covarep_feat = covarep_feat[:, :self.covarep_dim_ref]
                print(f"⚠️  COVAREP维度截断至：{self.covarep_dim_ref}")
            else:
                pad = torch.zeros((covarep_feat.shape[0], self.covarep_dim_ref - covarep_feat.shape[1]),
                                  device=covarep_feat.device)
                covarep_feat = torch.cat([covarep_feat, pad], dim=1)
                print(f"⚠️  COVAREP维度补0至：{self.covarep_dim_ref}")

        # ===== 核心修复2：确保注意力权重广播正确 =====
        # 计算注意力权重 (N, 2)
        attn_weights = self.attn_layer(x)
        # 拆分权重：HOG权重 (N,1)，COVAREP权重 (N,1)
        hog_w = attn_weights[:, 0:1]  # 取第一个权重，维度(N,1)
        covarep_w = attn_weights[:, 1:2]  # 取第二个权重，维度(N,1)

        # ===== 加权融合（广播乘法+拼接，而非相加！） =====
        # 关键修复：之前错误地将不同维度的张量相加，正确做法是「加权后拼接」
        hog_weighted = hog_w * hog_feat  # (N,4464)
        covarep_weighted = covarep_w * covarep_feat  # (N,63)
        fused_feat = torch.cat([hog_weighted, covarep_weighted], dim=1)  # (N,4527)

        # ===== GAT层前向 =====
        x = self.gat1(fused_feat, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # ===== 图池化+分类 =====
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out


# ===================== 6. 训练/评估函数 ======================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return correct / total if total > 0 else 0.0


# ===================== 7. 主程序（核心：动态获取输入维度） ======================
if __name__ == "__main__":
    # 加载被试列表
    train_subjs, dev_subjs, test_subjs = load_split_subjects()

    # 创建数据集
    try:
        train_dataset = DAICWOZDataset(train_subjs, split_type="train")
        dev_dataset = DAICWOZDataset(dev_subjs, split_type="dev")
        test_dataset = DAICWOZDataset(test_subjs, split_type="test")
    except Exception as e:
        print(f"创建数据集失败：{str(e)}")
        exit(1)

    # 检查数据集是否为空
    if len(train_dataset) == 0:
        print("❌ 训练集为空，无法训练")
        exit(1)

    # 获取实际的输入维度（从数据集中动态获取）
    input_dim = train_dataset.input_dim
    print(f"\n📌 模型输入维度：{input_dim}")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        num_workers=0
    )

    # 替换原模型初始化代码：
    model = AttentionGAT(
        input_dim=input_dim,  # 从数据集动态获取的总维度
        hog_dim_ref=config.HOG_DIM_REF,  # 4464（配置中的参考维度）
        covarep_dim_ref=config.COVAREP_DIM_REF,  # 63（配置中的参考维度）
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES
    )
    model = model.to(config.DEVICE)
    # 核心修改：weight_decay从1e-5→1e-4
    # 主程序中，替换原optimizer和加入学习率调度器：
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)  # lr从5e-4→8e-4
    # 学习率衰减：每10轮学习率×0.8
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = torch.nn.CrossEntropyLoss()

    # 打印信息
    print(f"\n训练集规模：{len(train_dataset)} 个图")
    print(f"验证集规模：{len(dev_dataset)} 个图")
    print(f"测试集规模：{len(test_dataset)} 个图")
    print(f"模型总参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)

    # 训练模型
    best_dev_acc = 0.0
    best_model_path = cfg.RESULTS_DIR + "attention_gat.pth"

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        dev_acc = evaluate(model, dev_loader, config.DEVICE)

        # 学习率衰减
        scheduler.step()

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc

            print("保存路径：", best_model_path)

            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch:02d} | 损失：{train_loss:.4f} | 验证集准确率：{dev_acc:.4f} → 保存模型")
        else:
            print(f"Epoch {epoch:02d} | 损失：{train_loss:.4f} | 验证集准确率：{dev_acc:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 测试集评估
    if os.path.exists(best_model_path):
        # 加载模型时确保维度匹配
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        test_acc = evaluate(model, test_loader, config.DEVICE)
        print("=" * 50)
        print(f"最终测试集准确率：{test_acc:.4f}")
        print(f"最优模型保存至：{best_model_path}")
    else:
        print("未保存任何模型")