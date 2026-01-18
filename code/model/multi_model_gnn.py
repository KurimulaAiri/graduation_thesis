import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import warnings

from code.utils.read_hog_files import Read_HOG_files

warnings.filterwarnings("ignore")


# ===================== 1. 数据配置 + GPU 设备设置 ======================
class Config:
    DATA_ROOT = r"D:\PRJ\pythonPRJ\graduation_thesis\data\AVEC2017"
    SPLIT_ROOT = r"D:\PRJ\pythonPRJ\graduation_thesis\data\AVEC2017"
    HOG_DIM = 4464
    COVAREP_DIM = 63
    HIDDEN_DIM = 128
    NUM_CLASSES = 4
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 0.001
    # ====== GPU 配置 ======
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 优先用GPU
    PIN_MEMORY = True  # 加速GPU数据传输（针对有GPU的情况）


config = Config()
# 打印设备信息
print(f"当前使用设备: {config.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")


# ===================== 2. 特征读取函数（无修改） ======================
def read_hog_file(hog_path):
    hog_dir = os.path.dirname(hog_path)
    subj_id = os.path.basename(hog_path).split('_')[0]
    user_name = f"{subj_id}_CLNF_hog.bin"

    hog_dict = Read_HOG_files(users=[user_name], hog_data_dir=hog_dir)
    if user_name not in hog_dict or hog_dict[user_name].size == 0:
        raise FileNotFoundError(f"HOG文件 {hog_path} 读取失败或为空")

    hog_data = hog_dict[user_name]
    if hog_data.shape[1] != config.HOG_DIM:
        raise ValueError(f"HOG特征维度错误（实际{hog_data.shape[1]}≠预期{config.HOG_DIM}）")

    print(f"成功读取HOG特征：{user_name} → 形状：{hog_data.shape}")
    return hog_data


def read_covarep_file(covarep_path):
    df = pd.read_csv(covarep_path, header=None)
    covarep_data = df.values.astype(np.float32)
    covarep_data = np.nan_to_num(covarep_data, nan=0.0, posinf=0.0, neginf=0.0)
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


# ===================== 3. 自定义数据集类（无修改） ======================
class DAICWOZDataset(Dataset):
    def __init__(self, subject_ids, split_type="train"):
        super().__init__()
        self.subject_ids = subject_ids
        self.split_type = split_type
        self.scaler = StandardScaler()
        self.data_list = self._build_dataset()

    def _build_dataset(self):
        data_list = []
        all_features = []

        print(f"\n当前处理的split类型：{self.split_type}")
        print(f"待处理的被试列表：{self.subject_ids[:5]}...（共{len(self.subject_ids)}个）")
        print(f"数据集根目录：{config.DATA_ROOT}")

        if self.split_type == "train":
            for subj in self.subject_ids:
                subj_num = subj.split('_')[0]
                hog_path = os.path.join(config.DATA_ROOT, subj, f"{subj_num}_CLNF_hog.bin")
                covarep_path = os.path.join(config.DATA_ROOT, subj, f"{subj_num}_COVAREP.csv")

                if not os.path.exists(hog_path) or not os.path.exists(covarep_path):
                    print(f"警告：{subj} 缺少特征文件，跳过")
                    continue

                try:
                    hog_data = read_hog_file(hog_path)
                    covarep_data = read_covarep_file(covarep_path)
                    hog_aligned, covarep_aligned = align_frames(hog_data, covarep_data)
                    fusion_features = np.hstack([hog_aligned, covarep_aligned])
                    all_features.append(fusion_features)
                except Exception as e:
                    print(f"读取 {subj} 特征失败：{str(e)}")
                    continue

            if len(all_features) == 0:
                raise ValueError("训练集未读取到任何有效特征！")
            self.scaler.fit(np.vstack(all_features))

        for subj in self.subject_ids:
            subj_num = subj.split('_')[0]
            hog_path = os.path.join(config.DATA_ROOT, subj, f"{subj_num}_CLNF_hog.bin")
            covarep_path = os.path.join(config.DATA_ROOT, subj, f"{subj_num}_COVAREP.csv")

            if not os.path.exists(hog_path) or not os.path.exists(covarep_path):
                continue

            try:
                hog_data = read_hog_file(hog_path)
                covarep_data = read_covarep_file(covarep_path)
                hog_aligned, covarep_aligned = align_frames(hog_data, covarep_data)

                fusion_features = np.hstack([hog_aligned, covarep_aligned])
                fusion_features = self.scaler.transform(fusion_features)
                num_frames = fusion_features.shape[0]

                split_file = os.path.join(config.SPLIT_ROOT, f"{self.split_type}_split_Depression_AVEC2017.csv")
                if self.split_type == "test":
                    split_file = os.path.join(config.SPLIT_ROOT, "full_test_split.csv")

                split_df = pd.read_csv(split_file)
                id_col = 'Participant_ID' if 'Participant_ID' in split_df.columns else 'participant_ID'
                phq8_score = split_df[split_df[id_col] == int(subj_num)]["PHQ8_Score"].values[0]
                label = phq8_to_level(phq8_score)

                x = torch.tensor(fusion_features, dtype=torch.float32)
                y = torch.tensor([label], dtype=torch.long)

                edge_index = []
                for t in range(num_frames - 1):
                    edge_index.append([t, t + 1])
                    edge_index.append([t + 1, t])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                if num_frames > 0 and edge_index.size(1) > 0:
                    data_list.append(Data(x=x, edge_index=edge_index, y=y, subject_id=subj))

            except Exception as e:
                print(f"处理 {subj} 失败：{str(e)}")
                continue

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ===================== 4. 加载数据集（无修改） ======================
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


# ===================== 5. GNN模型定义（无修改） ======================
class MultiModalGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=True, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False, dropout=0.3)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out


# ===================== 6. 训练与评估函数（核心修改：数据移到GPU） ======================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        # ====== 修改1：将batch数据移到指定设备 ======
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            # ====== 修改2：评估时数据也移到GPU ======
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
    return correct / total if total > 0 else 0.0


# ===================== 7. 主程序（核心修改：模型移到GPU + DataLoader优化） ======================
if __name__ == "__main__":
    train_subjs, dev_subjs, test_subjs = load_split_subjects()

    try:
        train_dataset = DAICWOZDataset(train_subjs, split_type="train")
        dev_dataset = DAICWOZDataset(dev_subjs, split_type="dev")
        test_dataset = DAICWOZDataset(test_subjs, split_type="test")
    except Exception as e:
        print(f"创建数据集失败：{str(e)}")
        exit(1)

    # ====== 修改3：DataLoader添加pin_memory（GPU加速） ======
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.PIN_MEMORY,  # 加速GPU数据传输
        num_workers=0  # 建议设为0，避免Windows下多进程问题
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

    # ====== 修改4：初始化模型并移到GPU ======
    input_dim = config.HOG_DIM + config.COVAREP_DIM
    model = MultiModalGAT(input_dim=input_dim, hidden_dim=config.HIDDEN_DIM, num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)  # 模型参数移到GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n训练集规模：{len(train_dataset)} 个图")
    print(f"验证集规模：{len(dev_dataset)} 个图")
    print(f"测试集规模：{len(test_dataset)} 个图")
    print(f"模型总参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50)

    best_dev_acc = 0.0
    best_model_path = "best_multi_modal_gat.pth"

    for epoch in range(1, config.EPOCHS + 1):
        # ====== 修改5：传入device参数 ======
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        dev_acc = evaluate(model, dev_loader, config.DEVICE)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch:02d} | 训练损失：{train_loss:.4f} | 验证集准确率：{dev_acc:.4f} → 保存模型")
        else:
            print(f"Epoch {epoch:02d} | 训练损失：{train_loss:.4f} | 验证集准确率：{dev_acc:.4f}")

    if os.path.exists(best_model_path):
        # ====== 修改6：加载模型时也移到GPU ======
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        test_acc = evaluate(model, test_loader, config.DEVICE)
        print("=" * 50)
        print(f"最终测试集准确率：{test_acc:.4f}")
        print(f"最优模型已保存至：{best_model_path}")
    else:
        print("未保存任何模型，跳过测试集评估")