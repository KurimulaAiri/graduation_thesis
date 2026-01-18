import torch
import os

import code.config as cfg


def test_torch_save(path):
    """
    测试 torch.save() 能否正常保存数据到指定路径
    Args:
        path: 目标保存路径（如 "./test_model.pth"）
    Returns:
        bool: 保存是否成功
    """
    # 步骤1：确保保存目录存在（如果路径包含多级目录）
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # 不存在则创建目录

    # 步骤2：创建测试数据（模拟模型/张量）
    test_tensor = torch.randn(3, 3)  # 简单张量
    test_model = torch.nn.Linear(10, 2)  # 简单模型（可选）

    try:
        # 测试1：保存张量
        torch.save(test_tensor, path)
        print(f"✅ 张量保存成功：{path}")

        # 验证1：检查文件是否存在
        if not os.path.exists(path):
            print("❌ 保存后文件不存在，路径可能异常")
            return False

        # 验证2：加载文件并检查数据完整性
        loaded_tensor = torch.load(path)
        if torch.allclose(test_tensor, loaded_tensor):
            print("✅ 保存的数据完整，加载后与原数据一致")
        else:
            print("❌ 加载后数据与原数据不一致")
            return False

        # 测试2（可选）：保存模型（state_dict）
        model_path = path.replace(".pth", "_model.pth")

        # model_path = cfg.RESULTS_DIR + "attention_gat.pth"

        torch.save(test_model.state_dict(), model_path)
        print(f"✅ 模型state_dict保存成功：{model_path}")

        # 清理临时文件（可选，避免冗余）
        os.remove(path)
        if os.path.exists(model_path):
            os.remove(model_path)

        return True

    except Exception as e:
        print(f"❌ torch.save 失败！错误信息：{e}")
        return False


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 测试路径1：当前目录（简单路径）
    test_path1 = "./test_save.pth"
    print("测试路径1：", test_path1)
    test_torch_save(test_path1)

    # 测试路径2：多级目录（验证自动创建目录）
    test_path2 = cfg.RESULTS_DIR + "test_save.pth"
    print("\n测试路径2：", test_path2)
    test_torch_save(test_path2)

    # 测试路径3：权限不足的路径（如系统目录，会报错）
    # test_path3 = "/root/test_save.pth"  # 非root用户运行会失败
    # print("\n测试路径3：", test_path3)
    # test_torch_save(test_path3)