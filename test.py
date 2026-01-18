# import torch
# print("CUDA可用:", torch.cuda.is_available())  # 输出True则成功
# print("PyTorch版本:", torch.__version__)       # 应输出2.9.0
# print("CUDA版本:", torch.version.cuda)         # 应输出12.6


import code.config as cfg
import numpy as np
from code.utils import Read_HOG_files

# 示例调用（对应runHOGread_example.m）
if __name__ == "__main__":
    users = ['322.CLM_hog.bin']
    hog_data_dir = cfg.DATA_ROOT + '322_P\\'

    hog_data, valid_inds, vid_id = Read_HOG_files(users, hog_data_dir)

    # 可选保存为.npy（替代MATLAB的.mat）
    np.savez(cfg.RESULTS_DIR + '322.CLM_hog.npz', hog_data=hog_data, valid_inds=valid_inds, vid_id=vid_id)
