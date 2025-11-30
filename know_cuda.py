import torch

if torch.cuda.is_available():
    idx = 0  # 你想用的显卡编号
    print(f"当前使用的显卡索引: {idx}")
    print(f"显卡名称: {torch.cuda.get_device_name(idx)}")

    # 简单的显存测试
    x = torch.randn(1000, 1000).cuda(idx)
    print("成功在 RTX 3090 上创建了 Tensor")
else:
    print("未检测到 GPU")