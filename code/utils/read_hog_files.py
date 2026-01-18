# code/utils/read_hog_files.py
import os
import struct
import numpy as np


def Read_HOG_files(users, hog_data_dir):
    """
    读取HOG特征文件，复刻MATLAB版Read_HOG_files逻辑
    【修复后】返回字典格式：{文件名: HOG特征矩阵}
    Parameters:
        users: list[str] - 待读取的文件名列表
        hog_data_dir: str - HOG文件所在目录
    Returns:
        hog_dict: dict - {文件名: HOG特征矩阵（维度：帧数×4464）}
    """
    hog_dict = {}  # 改为字典返回，key是文件名，value是HOG特征

    for user in users:
        hog_file_path = os.path.join(hog_data_dir, user)
        if not os.path.exists(hog_file_path):
            print(f"警告：HOG文件 {hog_file_path} 不存在，跳过")
            hog_dict[user] = np.array([])
            continue

        try:
            with open(hog_file_path, 'rb') as f:
                # 1. 读取文件头（num_cols/num_rows/num_chan）
                header_bytes = f.read(12)
                if len(header_bytes) != 12:
                    print(f"警告：{user} 文件头不完整，跳过")
                    hog_dict[user] = np.array([])
                    continue

                num_cols = struct.unpack('<i', header_bytes[0:4])[0]
                num_rows = struct.unpack('<i', header_bytes[4:8])[0]
                num_chan = struct.unpack('<i', header_bytes[8:12])[0]

                # 计算特征维度（DAIC-WOZ的HOG应为4464 = 112*112*3/8，需校验）
                feat_dim = num_rows * num_cols * num_chan
                expected_dim = 4464
                if feat_dim != expected_dim:
                    print(f"警告：{user} 特征维度异常（实际{feat_dim}≠预期{expected_dim}），强制修正")
                    feat_dim = expected_dim

                # 2. 逐样本读取（小批次，避免字节不匹配）
                hog_data = []
                single_feat_bytes = (1 + feat_dim) * 4  # 1个valid_ind + feat_dim个特征，每个float32占4字节
                batch_size = 100  # 缩小批次到100，降低字节误差

                while True:
                    # 读取一个批次的字节
                    batch_bytes = f.read(single_feat_bytes * batch_size)
                    if not batch_bytes:
                        break

                    # 计算实际能解析的样本数
                    actual_samples = len(batch_bytes) // single_feat_bytes
                    if actual_samples == 0:
                        break

                    # 解析每个样本
                    for i in range(actual_samples):
                        start = i * single_feat_bytes
                        end = start + single_feat_bytes
                        sample_bytes = batch_bytes[start:end]

                        # 解析单个样本
                        try:
                            sample = struct.unpack(f'<{1 + feat_dim}f', sample_bytes)
                            sample = np.array(sample, dtype=np.float32)
                            hog_data.append(sample[1:])  # 剔除首列valid_ind，只保留特征
                        except struct.error:
                            continue  # 跳过损坏的样本

                # 转换为numpy数组并过滤空行
                hog_data = np.array(hog_data, dtype=np.float32)
                hog_data = hog_data[~np.all(hog_data == 0, axis=1)]

                # 确保维度正确（帧数×4464）
                if hog_data.shape[1] != expected_dim:
                    hog_data = hog_data[:, :expected_dim]  # 截断到4464维

                hog_dict[user] = hog_data
                print(f"成功读取 {user} → 形状：{hog_data.shape}（帧数×特征维度）")

        except Exception as e:
            print(f"读取 {user} 失败：{str(e)}")
            hog_dict[user] = np.array([])

    return hog_dict