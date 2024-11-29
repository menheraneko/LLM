import torch

# 3个训练示例
idx = torch.tensor([2, 3, 1])

# 获取最大 token ID 加 1 来确定嵌入矩阵行数
num_idx = max(idx) + 1

# 嵌入维度5
out_dim = 5

# 设置种子
torch.manual_seed(123)

# 创建嵌入层
embedding = torch.nn.Embedding(num_idx, out_dim)

# 获取token id 1的嵌入向量
embedding(torch.tensor([1]))

# 获取token id 2的嵌入向量
embedding(torch.tensor([2]))

# 类似地定义待嵌入向量
idx = torch.tensor([2, 3, 1])

# 嵌入
embedding(idx)




# 使用 one-hot 编码表示token ID
onehot = torch.nn.functional.one_hot(idx)

# 设置种子
torch.manual_seed(123)

# 创建线性层，输入维度为token数，输出维度一致，不加入bias
linear = torch.nn.Linear(num_idx, out_dim, bias=False)

# 将线性层的权重设置为嵌入层权重的转置
linear.weight = torch.nn.Parameter(embedding.weight.T)

# 线性变换
linear(onehot.float())

# 嵌入
embedding(idx)
