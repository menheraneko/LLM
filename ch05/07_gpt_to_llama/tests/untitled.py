import torch  # 导入 PyTorch 库

theta_base = 10_000  # 设置一个基础值 θ_base

# 遍历头维度从 1 到 11
for head_dim in range(1, 12):
    # 计算 before
    # 生成一个从 0 到 head_dim // 2 的整数序列，然后通过公式计算结果
    before = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    # 计算 after
    # 生成一个从 0 到 head_dim 的偶数序列，并将其转换为浮点数，然后通过公式计算结果
    after = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # 检查 before 和 after 是否相等，并格式化输出字符串
    s = f"{torch.equal(before, after)} | head dim: {head_dim}, {before}, {after}"
    print(s)  # 打印结果字符串