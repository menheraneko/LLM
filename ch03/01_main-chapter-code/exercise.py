import torch
import sys
import os
import torch.nn as nn

parent_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件夹的绝对路径
sys.path.append(parent_dir)  # 将父目录添加到路径中


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False): # 初始化，额外增加上下文长度和dropout率
        super().__init__()
        self.d_out = d_out # 输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # query线性层
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias) # key
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # value
        self.dropout = nn.Dropout(dropout) # dropout层
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # 保存缓冲数据，名称mask，数据为对应的dropout中间矩阵

    def forward(self, x): # 前向传播
        b, num_tokens, d_in = x.shape # batch size，token数量，嵌入维度
        keys = self.W_key(x) # 计算key
        queries = self.W_query(x) # 计算query
        values = self.W_value(x) # 计算value

        attn_scores = queries @ keys.transpose(1, 2) # 计算注意力分数。使用transpose交换1，2维度，即转置
        attn_scores.masked_fill_(  # 掩码
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # 取到num_tokens列和行进行mask操作。填充负无穷
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        ) # 归一化
        attn_weights = self.dropout(attn_weights) # dropout

        context_vec = attn_weights @ values # 上下文向量
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): # 初始化，加入了head数量的要求
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        ) # 多个单头注意力的列表，共num_head个

    def forward(self, x): # 前向传播
        return torch.cat([head(x) for head in self.heads], dim=-1) # cat操作，得到多通道并行的多头注意力，在最后一个维度拼接（特征维度）


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): # 初始化
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out # dropout率
        self.num_heads = num_heads # 头数
        self.head_dim = d_out // num_heads  # 保证输出任然为d_out维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # query
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) # key
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # value
        self.out_proj = nn.Linear(d_out, d_out)  # 多头注意力整合层
        self.dropout = nn.Dropout(dropout) # dropout层
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        ) # 此处和上面同理，保存中间矩阵

    def forward(self, x): # 前向传播
        b, num_tokens, d_in = x.shape # batch size，token数量，嵌入维度

        keys = self.W_key(x)  # key
        queries = self.W_query(x) # query
        values = self.W_value(x) # value


        # 展开权重的最后一个维度: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 此处是便于为所有头提供权重支持
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 掩码填充
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1) # 最后一个维度归一化
        attn_weights = self.dropout(attn_weights) # dropout

        context_vec = (attn_weights @ values).transpose(1, 2) # 计算上下文向量，并恢复形状

        # self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # 重塑形状，同时使用contiguous保证内存连续
        context_vec = self.out_proj(context_vec)  # 整合多头注意力

        return context_vec


# 3.1
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in, d_out = 3, 2 # 输入，输出维度

# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123) # 设置种子
sa_v1 = SelfAttention_v1(d_in, d_out) # 获取结果

# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123) # 设置种子
sa_v2 = SelfAttention_v2(d_in, d_out) # 获取上下文结果

# 权重矩阵
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T) # query
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T) # key
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T) # value

sa_v1(inputs) # 分别进行处理
sa_v2(inputs)








# 3.2
torch.manual_seed(123)


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0) # 模拟批次数据，batch size为2
context_length = batch.shape[1] # 上下文长度
d_out = 1 # 输出维度
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2) # 创建对象

context_vecs = mha(batch) # 上下文向量

print(context_vecs) # 输出结果
print("context_vecs.shape:", context_vecs.shape) # 形状



# 3.3
context_length = 1024 # 上下文长度
d_in, d_out = 768, 768 # 输入和输出维度
num_heads = 12 # 头数

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads) # 创建对象

def count_parameters(model): # 计算参数数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # 若需要梯度传播，则统计

count_parameters(mha) # 参数数量