import torch
import torch.nn as nn



# 3.3.1部分代码

# 测试向量表示
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


query = inputs[1]  # 第二个input token为query

attn_scores_2 = torch.empty(inputs.shape[0]) # 初始化注意力分数
for i, x_i in enumerate(inputs): # 逐token计算得分
    attn_scores_2[i] = torch.dot(x_i, query) # 此处为dim=1，无需转置

print(attn_scores_2)

res = 0.

for idx, element in enumerate(inputs[0]): # 一个计算[0]和[1]元素的例子
    res += inputs[0][idx] * query[idx] # 点积累加

print(res)
print(torch.dot(inputs[0], query))



attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() # 计算注意力权重，使用单个得分/总和，保证sum为1

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

def softmax_naive(x): # 归一化
    return torch.exp(x) / torch.exp(x).sum(dim=0) # 归一化函数，在第0维上进行。

attn_weights_2_naive = softmax_naive(attn_scores_2) # 归一化得分

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())


attn_weights_2 = torch.softmax(attn_scores_2, dim=0) # 直接调用库函数

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1] # 测试用query，和上面同理

context_vec_2 = torch.zeros(query.shape) # 初始化上下文vector全为0
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i # 注意力weight*input得到基于第二个query的上下文 vector

print(context_vec_2) # 输出结果







#3.3.2部分代码
attn_scores = torch.empty(6, 6) # 初始化attention分数

for i, x_i in enumerate(inputs): # 双层for，计算所有token的attention分数
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j) # 点积运算

print(attn_scores) # 输出结果


attn_scores = inputs @ inputs.T #input矩阵乘以自身转置得到attention分数
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1) # 归一化处理，在列上进行。
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]) # 任意取一列，验证是否归一化成功
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1)) # 求和

all_context_vecs = attn_weights @ inputs # weight*input得到上下文vector
print(all_context_vecs) # 输出上下文vector
print("Previous 2nd context vector:", context_vec_2) # 之前的测试用例





#3.4.1部分代码
x_2 = inputs[1] # 第二个测试用例
d_in = inputs.shape[1] # 输入dim，这里为3
d_out = 2 #输出dim为2

torch.manual_seed(123) # 设置种子

# 初始化权重矩阵，不使用梯度更新
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # query
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # key
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # value

query_2 = x_2 @ W_query # 计算query向量
key_2 = x_2 @ W_key # 计算key
value_2 = x_2 @ W_value # 计算value
print(query_2) # 输出query向量

keys = inputs @ W_key # input得到的key向量
values = inputs @ W_value # 得到value

print("keys.shape:", keys.shape) # 输出，这里应当dim为2
print("values.shape:", values.shape)

keys_2 = keys[1] # 取第二个key作为测试
attn_score_22 = query_2.dot(keys_2) # 计算点积，得到query的attention分数
print(attn_score_22) # 输出注意力分数

attn_scores_2 = query_2 @ keys.T # 计算全部query的attention分数
print(attn_scores_2) # 输出

d_k = keys.shape[1] # 测试用
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # 归一化，得到注意力权重。通过除以dim的0.5次方来缩放
print(attn_weights_2) # 输出权重

context_vec_2 = attn_weights_2 @ values # 计算上下文向量
print(context_vec_2) # 输出





#3.4.2部分代码
# self-attention模块
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out): # 初始化函数，传入输入嵌入维度和输出嵌入维度
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) # 根据维度需求初始化权重矩阵——query
        self.W_key = nn.Parameter(torch.rand(d_in, d_out)) # key
        self.W_value = nn.Parameter(torch.rand(d_in, d_out)) # value

    def forward(self, x): # 前向传播计算
        keys = x @ self.W_key # 计算key
        queries = x @ self.W_query # 计算query
        values = x @ self.W_value # 计算value

        attn_scores = queries @ keys.T  # 得到input的总注意力分数
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1 # 归一化注意力分数，获取注意力权重。缩放同上
        )

        context_vec = attn_weights @ values # 计算上下文向量
        return context_vec


torch.manual_seed(123) # 设置种子
sa_v1 = SelfAttention_v1(d_in, d_out) # 获取attention处理模块对象
print(sa_v1(inputs)) # 输出处理结果

# 第二版本，使用线性层简化结构
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False): # 禁用bias，使其变成矩阵乘法
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # 线形层模拟权重query，相当于直接用线形层权重进行了替换
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) # key
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # value

    def forward(self, x): # 前向传播
        keys = self.W_key(x) # 计算key
        queries = self.W_query(x) # 计算query
        values = self.W_value(x) # 计算value

        attn_scores = queries @ keys.T # 计算attention分数
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1) # 归一化并缩放

        context_vec = attn_weights @ values # 计算上下文向量
        return context_vec


torch.manual_seed(789) # 设置种子
sa_v2 = SelfAttention_v2(d_in, d_out) # 获取处理对象
print(sa_v2(inputs)) # 输出处理结果





#3.5.1部分代码
queries = sa_v2.W_query(inputs) # 直接使用上一节的权重
keys = sa_v2.W_key(inputs) # key
attn_scores = queries @ keys.T # 得到注意力分数

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # 归一化缩放
print(attn_weights) # 输出注意力权重

context_length = attn_scores.shape[0] # 的上下文向量长度
mask_simple = torch.tril(torch.ones(context_length, context_length)) # 获取下三角矩阵，也就是生成一个下三角为1，上三角为0的(n,n)矩阵。
print(mask_simple) # 输出

masked_simple = attn_weights*mask_simple # 上三角为0，计算后变为0无效化，达到遮掩的效果。
print(masked_simple) # 输出计算结果

row_sums = masked_simple.sum(dim=-1, keepdim=True) # 逐项求和，类似地获取列的sum
masked_simple_norm = masked_simple / row_sums # 保证每列之和仍然是1
print(masked_simple_norm) # 输出结果

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) # 获取上三角矩阵
masked = attn_scores.masked_fill(mask.bool(), -torch.inf) # 通过bool判断是否为需要掩码的部分，填充无穷。
print(masked) # 输出结果

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1) # 归一化，负无穷趋近于0
print(attn_weights) # 输出注意力权重









#3.5.2部分代码
torch.manual_seed(123) # 设置种子
dropout = torch.nn.Dropout(0.5) # 设置dropout层，drop rate为0.5
example = torch.ones(6, 6) # 创建示例

print(dropout(example)) # dropout

torch.manual_seed(123) # 设置种子
print(dropout(attn_weights)) # 随机drop注意力权重




#3.5.3部分代码
batch = torch.stack((inputs, inputs), dim=0) # 模拟批次数据，batch size为2
print(batch.shape) # 输出结果

# 因果注意力处理模块
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

torch.manual_seed(123) # 设置种子

context_length = batch.shape[1] # 上下文长度
ca = CausalAttention(d_in, d_out, context_length, 0.0) # 获取处理对象

context_vecs = ca(batch) # 计算上下文向量

print(context_vecs) # 输出结果
print("context_vecs.shape:", context_vecs.shape) # 输出形状












#3.6.1部分代码
# 简单多头注意力处理器
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False): # 初始化，加入了head数量的要求
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        ) # 多个单头注意力的列表，共num_head个

    def forward(self, x): # 前向传播
        return torch.cat([head(x) for head in self.heads], dim=-1) # cat操作，得到多通道并行的多头注意力，在最后一个维度拼接（特征维度）


torch.manual_seed(123) # 设置种子

context_length = batch.shape[1] # token数量
d_in, d_out = 3, 2 # 输入和输出维度
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
) # 多头注意力实例化

context_vecs = mha(batch) # 获得多头注意力的上下文向量

print(context_vecs) # 输出结果
print("context_vecs.shape:", context_vecs.shape) # 输出形状，cat之后，特征维度从2变成2*2了。因为有2个头








#3.6.2部分代码
# 多头注意力
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


torch.manual_seed(123) # 设置种子

batch_size, context_length, d_in = batch.shape # 获取参数
d_out = 2 # 输出维度
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2) # 获取对象

context_vecs = mha(batch) # 获取多头注意力的上下文向量

print(context_vecs) # 输出结果
print("context_vecs.shape:", context_vecs.shape) # 形状


a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]]) # 测试用例，展示queries @ keys.transpose(2, 3)的细节

print(a @ a.transpose(2, 3)) # 首先，会把key进行转置

first_head = a[0, 0, :, :] # 这是第一个头
first_res = first_head @ first_head.T # 第一个头部分的计算
print("First head:\n", first_res) # 结果

second_head = a[0, 1, :, :] # 第二个头
second_res = second_head @ second_head.T # 同理
print("\nSecond head:\n", second_res) # 结果