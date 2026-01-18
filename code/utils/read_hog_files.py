# code/utils/read_hog_files.py
import os
import struct
import numpy as np
import traceback
import gc


def Read_HOG_files(users, hog_data_dir, downsample_rate=10, use_float16=True, max_memory_mb=200):
    """
    极致内存优化版HOG读取
    Parameters:
        users: list[str] - 文件名列表
        hog_data_dir: str - 文件目录
        downsample_rate: int - 下采样率（每隔10帧取1帧）
        use_float16: bool - 使用float16
        max_memory_mb: int - 单文件最大内存限制（超过则进一步降采样）
    Returns:
        hog_dict: dict - {文件名: HOG特征矩阵}
    """
    hog_dict = {}
    expected_dim = 4464
    dtype = np.float16 if use_float16 else np.float32
    bytes_per_elem = 2 if use_float16 else 4

    for user in users:
        hog_file_path = os.path.join(hog_data_dir, user)
        if not os.path.exists(hog_file_path):
            print(f"⚠️  文件不存在：{user}")
            hog_dict[user] = np.array([])
            continue

        try:
            with open(hog_file_path, 'rb') as f:
                # 读取文件头
                header_bytes = f.read(12)
                if len(header_bytes) != 12:
                    print(f"⚠️  文件头损坏：{user}")
                    hog_dict[user] = np.array([])
                    continue

                # 逐帧读取+极致下采样
                hog_data = []
                single_feat_bytes = (1 + expected_dim) * 4
                frame_count = 0
                effective_rate = downsample_rate  # 动态调整下采样率

                while True:
                    # 单帧读取（彻底避免批次内存占用）
                    frame_bytes = f.read(single_feat_bytes)
                    if not frame_bytes:
                        break

                    frame_count += 1
                    # 动态下采样：先按基础率，再按内存限制二次采样
                    if frame_count % effective_rate != 0:
                        continue

                    # 解析单帧
                    try:
                        sample = struct.unpack(f'<{1 + expected_dim}f', frame_bytes)
                        sample = np.array(sample[1:], dtype=dtype)
                        hog_data.append(sample)
                    except struct.error:
                        continue

                    # 内存超限检查：提前终止，避免堆积
                    if len(hog_data) * expected_dim * bytes_per_elem > max_memory_mb * 1024 * 1024:
                        effective_rate *= 2  # 加倍下采样率
                        print(f"⚠️  {user} 内存超限，下采样率提升至{effective_rate}")
                        # 清空当前数据，重新读取（用新的采样率）
                        hog_data = []
                        frame_count = 0
                        f.seek(12)  # 回到文件头，重新读取
                        continue

                # 转换为数组并过滤空行
                hog_data = np.array(hog_data, dtype=dtype)
                hog_data = hog_data[~np.all(hog_data == 0, axis=1)]

                # 强制维度对齐
                if hog_data.shape[1] != expected_dim:
                    pad_width = ((0, 0), (0, expected_dim - hog_data.shape[1])) if hog_data.shape[
                                                                                       1] < expected_dim else ((0, 0),
                                                                                                               (0, 0))
                    hog_data = np.pad(hog_data, pad_width, mode='constant')[:, :expected_dim]

                # 最终内存检查
                final_memory_mb = hog_data.nbytes / 1024 / 1024
                print(f"✅  {user} → 原始帧={frame_count} → 采样后帧={hog_data.shape[0]} → 内存={final_memory_mb:.1f}MB")

                hog_dict[user] = hog_data
                gc.collect()  # 强制垃圾回收

        except MemoryError:
            print(f"❌  内存不足：{user}")
            hog_dict[user] = np.array([])
            gc.collect()
        except Exception as e:
            print(f"❌  读取失败：{user} - {str(e)}")
            traceback.print_exc()
            hog_dict[user] = np.array([])
            gc.collect()

    return hog_dict