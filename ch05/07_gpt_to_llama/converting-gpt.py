import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块





# 1.1

# class LayerNorm(nn.Module):  # 注释掉的 LayerNorm 类
# def __init__(self, emb_dim):  # 初始化函数，输入的参数是 emb_dim
# super().__init__()  # 调用父类的初始化函数
# self.eps = 1e-5  # 设置一个小的数值用于数值稳定性
# self.scale = nn.Parameter(torch.ones(emb_dim))  # 初始化 scale 参数，大小为 emb_dim，值为 1
#  self.shift = nn.Parameter(torch.zeros(emb_dim))  # 初始化 shift 参数，大小为 emb_dim，值为 0

# def forward(self, x):  # 前向传播函数，输入 x 是输入张量
# mean = x.mean(dim=-1, keepdim=True)  # 计算 x 的均值，沿着最后一个维度进行
# var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算 x 的方差，沿着最后一个维度进行
# norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 对 x 进行标准化
#  return self.scale * norm_x + self.shift  # 对标准化后的 x 进行缩放和平移操作，返回结果

# 定义一个 RMSNorm 类，继承自 nn.Module
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        # 初始化 RMSNorm 类，emb_dim 是嵌入维度，eps 是防止除零的极小值
        super().__init__()  # 调用父类的初始化方法

        self.eps = eps  # 保存 epsilon 值
        self.emb_dim = emb_dim  # 保存嵌入维度
        # 创建一个可学习的权重参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        # 正向传播方法，x 是输入的张量

        # 计算 x 的均方值（即每个特征的均值的平方），在最后一个维度上求平均
        means = x.pow(2).mean(dim=-1, keepdim=True)

        # 通过均方根进行规范化，避免除零问题
        x_normed = x * torch.rsqrt(means + self.eps)

        # 将规范化后的结果乘以可学习的权重，并转换为与输入相同的数据类型
        return (x_normed * self.weight).to(dtype=x.dtype)

# 种子
torch.manual_seed(123)

# 创建一个形状为 (2, 3, 4) 的随机张量，模拟一个批量的输入数据
example_batch = torch.randn(2, 3, 4)

# 实例化自定义的 RMSNorm 层，嵌入维度为 example_batch 最后一维的大小
rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])

# 实例化 PyTorch 提供的 RMSNorm 层，嵌入维度为 example_batch 最后一维的大小
rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

# 比较自定义的 RMSNorm 层和 PyTorch 提供的 RMSNorm 层的输出是否相同
assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))  # 如果输出不相同会抛出错误






# 1.2


# class GELU(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(
#             torch.sqrt(torch.tensor(2.0 / torch.pi)) *
#             (x + 0.044715 * torch.pow(x, 3))
#         ))
# 这一部分代码被注释掉了，定义了一个 GELU 激活函数类。GELU（Gaussian Error Linear Unit）是一种常用的激活函数。

class SiLU(nn.Module):
    # 定义一个新的类 SiLU，继承自 nn.Module 类，表示 SiLU 激活函数
    def __init__(self):
        # 初始化方法，调用父类的构造函数
        super(SiLU, self).__init__()

    def forward(self, x):
        # 定义前向传播过程，即 SiLU 激活函数的计算方式
        # SiLU 是 x * sigmoid(x)
        return x * torch.sigmoid(x)

# 创建 SiLU 类的一个实例
silu = SiLU()

# 使用 assert 来检查 SiLU 函数的实现是否与 PyTorch 中的内置功能 torch.nn.functional.silu 一致
# example_batch 应该是一个已经定义的 tensor 输入
# torch.allclose 用于判断两个张量在数值上是否相等
assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))

# 如果 SiLU 类的实现和 PyTorch 内置的 SiLU 函数的结果一致，assert 不会抛出错误。








# 1.3
class FeedForward(nn.Module):  # 定义一个继承自nn.Module的FeedForward类
    def __init__(self, cfg):  # 构造函数，cfg为配置字典，包含模型参数
        super().__init__()  # 调用父类nn.Module的构造函数
        # 定义三个线性层（全连接层）和一个激活函数
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  # 第一个全连接层，将输入维度从emb_dim变换到hidden_dim，且不使用偏置项
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  # 第二个全连接层，同样的输入输出维度，且不使用偏置项
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)  # 第三个全连接层，将隐藏层的维度从hidden_dim映射回emb_dim
        self.silu = SiLU()  # 定义SiLU激活函数（Sigmoid Linear Unit），类似ReLU，但带有sigmoid的平滑性质

    def forward(self, x):  # 定义前向传播函数
        x_fc1 = self.fc1(x)  # 通过第一个全连接层得到x_fc1
        x_fc2 = self.fc2(x)  # 通过第二个全连接层得到x_fc2
        x = self.silu(x_fc1) * x_fc2  # 使用SiLU激活函数对x_fc1进行激活，并与x_fc2逐元素相乘
        return self.fc3(x)  # 将结果输入到第三个全连接层，输出最终结果








# 1.4
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    # 检查头部维度是否为偶数，确保可以分成两半
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算逆频率（inv_freq），用于后续的位置编码
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # 生成位置索引（0 到 context_length-1）
    positions = torch.arange(context_length)

    # 计算角度（每个位置的角度），形状为 (context_length, head_dim // 2)
    angles = positions[:, None] * inv_freq[None, :]

    # 扩展角度以匹配 head_dim（复制一份角度数据，变为 (context_length, head_dim)）
    angles = torch.cat([angles, angles], dim=1)

    # 计算正弦和余弦值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 返回预计算的正弦和余弦
    return cos, sin

def compute_rope(x, cos, sin):
    # x: 输入张量，形状为 (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    # 检查 head_dim 是否为偶数
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 将 x 拆分为前半部分和后半部分，形状为 (batch_size, num_heads, seq_len, head_dim // 2)
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2 :]  # 后半部分

    # 调整 cos 和 sin 的形状以匹配输入 x 的长度和头部维度
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转位置编码：对后半部分 (x2) 进行旋转，得到 (x1, -x2)
    rotated = torch.cat((-x2, x1), dim=-1)

    # 计算最终的旋转后的输入：x1 和 -x2 与 cos 和 sin 相乘后相加
    x_rotated = (x * cos) + (rotated * sin)

    # 返回旋转后的张量，类型与原始输入相同
    return x_rotated.to(dtype=x.dtype)

# 设置一些参数
batch_size = 2  # 批次大小
context_len = 5  # 序列长度
num_heads = 4  # 注意力头的数量
head_dim = 16  # 每个头的维度

# 预计算 RoPE 的正弦和余弦值
cos, sin = precompute_rope_params(head_dim=head_dim, context_length=context_len)

# 创建随机的查询（queries）和键（keys）张量，模拟注意力计算中的输入
torch.manual_seed(123)  # 设置随机种子以确保可重复性
queries = torch.randn(batch_size, num_heads, context_len, head_dim)  # 随机生成查询张量
keys = torch.randn(batch_size, num_heads, context_len, head_dim)  # 随机生成键张量

# 将旋转位置编码应用到查询和键张量
queries_rot = compute_rope(queries, cos, sin)  # 对查询应用 RoPE
keys_rot = compute_rope(keys, cos, sin)  # 对键应用 RoPE







# 1.5
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        # 初始化 MultiHeadAttention 类，参数包括输入维度 d_in、输出维度 d_out、上下文长度、头的数量以及数据类型
        super().__init__()  # 调用父类的初始化方法

        # 确保输出维度可被头的数量整除
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out  # 保存输出维度
        self.num_heads = num_heads  # 保存头的数量
        self.head_dim = d_out // num_heads  # 计算每个头的维度

        ################################### NEW ###################################
        # 所有线性层的偏置设置为 False，数据类型为 dtype
        ###########################################################################
        # 创建查询、键和值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # 创建输出层，用于合并头的输出
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        # self.dropout = nn.Dropout(dropout)  # 可选：添加 dropout 层
        # 创建一个上三角矩阵作为缓冲区，以便于计算掩码
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        ################################### NEW ###################################
        # 预计算旋转编码的正弦和余弦参数，并将其注册为缓冲区
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        ###########################################################################

    def forward(self, x):
        # 正向传播方法，x 是输入张量，形状为 (b, num_tokens, d_in)

        b, num_tokens, d_in = x.shape  # 获取批次大小、token 数量和输入维度

        # 通过线性层获取键、查询和值的表示，形状变为 (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过增加一个头的维度隐式拆分矩阵
        # 将最后一个维度展开：形状变为 (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：将 (b, num_tokens, num_heads, head_dim) 转换为 (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        ################################### NEW ###################################
        # 计算旋转编码，并应用于键和查询
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)
        ###########################################################################

        # 计算带有因果掩码的缩放点积注意力（自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 对每个头进行点积

        # 将原始掩码截断到 token 数量并转换为布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重，进行 softmax 归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # attn_weights = self.dropout(attn_weights)  # 可选：应用 dropout（如果有定义）

        # 形状为 (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并头的输出，self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 对输出进行线性变换（可选）

        return context_vec  # 返回上下文向量
# 设置参数
batch_size = 1  # 批次大小
context_len = 100  # 上下文长度
max_context_len = 4096  # 最大上下文长度
embed_dim = 128  # 嵌入维度
num_heads = 4  # 注意力头的数量

# 创建一个随机输入批次，形状为 (batch_size, context_len, embed_dim)
example_batch = torch.randn((batch_size, context_len, embed_dim))

# 初始化 MultiHeadAttention 模块
mha = MultiHeadAttention(
    d_in=embed_dim,            # 输入维度
    d_out=embed_dim,           # 输出维度
    context_length=max_context_len,  # 上下文长度设置为最大值
    num_heads=num_heads         # 设置注意力头数量
)

# 通过多头注意力模块进行前向传播，传入随机生成的输入批次
output = mha(example_batch)

# 删除 mha 对象，以释放内存
del mha








# 1.6
class TransformerBlock(nn.Module):
    # 初始化方法，接受配置字典作为参数
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头自注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入维度
            d_out=cfg["emb_dim"],  # 输出维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],  # 注意力头的数量
            dtype=cfg["dtype"]  # 数据类型（例如 float32）
        )
        # 初始化前馈网络
        self.ff = FeedForward(cfg)

        ################################### NEW ###################################
        # 使用RMSNorm代替LayerNorm进行归一化
        self.norm1 = RMSNorm(cfg["emb_dim"])  # 第一层归一化
        self.norm2 = RMSNorm(cfg["emb_dim"])  # 第二层归一化
        ###########################################################################

        # self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 可能是用于丢弃短接连接的dropout（暂时注释掉）

    # 前向传播方法
    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x  # 保存输入x，后续与输出相加
        x = self.norm1(x)  # 对输入x进行归一化
        x = self.att(x)    # 通过多头自注意力层进行处理，输出形状为 [batch_size, num_tokens, emb_size]
        # x = self.drop_shortcut(x)  # 可选的短接丢弃（暂时注释掉）
        x = x + shortcut  # 将输入x与输出相加，形成残差连接

        # 前馈网络块的快捷连接
        shortcut = x  # 保存经过自注意力后的结果，后续与前馈网络输出相加
        x = self.norm2(x)  # 对x进行第二次归一化
        x = self.ff(x)  # 通过前馈网络处理x
        # x = self.drop_shortcut(x)  # 可选的短接丢弃（暂时注释掉）
        x = x + shortcut  # 将输入x与前馈网络输出相加，形成残差连接

        return x  # 返回最终输出




# 1.7
# class GPTModel(nn.Module):
class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        # self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        ################################### NEW ###################################
        # self.final_norm = LayerNorm(cfg["emb_dim"])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        ###########################################################################
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds  # + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表的大小，即模型能够识别的词汇数量
    "context_length": 1024,  # 上下文长度，表示模型能够处理的最大输入序列长度
    "emb_dim": 768,          # 嵌入层的维度，表示每个词的向量表示的大小
    "n_heads": 12,           # 自注意力机制的头数，指的是并行计算的注意力头数
    "n_layers": 12,          # 模型的层数，即Transformer中的堆叠层数
    "drop_rate": 0.1,        # Dropout率，用于防止过拟合的正则化技术，指在训练时丢弃神经网络的一部分连接
    "qkv_bias": False        # 是否使用Query、Key、Value的偏置项，通常用于控制自注意力机制的计算
}





2
# 定义一个字典，用于存储GPT模型配置参数（124M版本）
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表的大小
    "context_length": 1024,  # 上下文长度（即模型每次可以处理的最大token数）
    "emb_dim": 768,          # 嵌入维度（词向量的维度）
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # Transformer层的数量
    "drop_rate": 0.1,        # Dropout的概率（防止过拟合）
    "qkv_bias": False        # 是否启用Query、Key、Value的偏置项
}

# 定义另一个字典，用于存储更大规模GPT模型配置参数（1558M版本）
GPT_CONFIG_1558M = {
    "vocab_size": 50257,     # 词汇表的大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 1600,         # 嵌入维度
    "n_heads": 25,           # 注意力头的数量
    "n_layers": 48,          # Transformer层的数量
    "drop_rate": 0.1,        # Dropout的概率
    "qkv_bias": False        # 是否启用Query、Key、Value的偏置项
}

# 定义LLAMA2模型的配置（7B版本）
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # 词汇表的大小
    "context_length": 4096,  # 上下文长度
    "emb_dim": 4096,         # 嵌入维度
    "n_heads": 32,           # 注意力头的数量
    "n_layers": 32,          # Transformer层的数量
    "hidden_dim": 11008,     # FeedForward层的隐藏维度（新的配置项）
    "dtype": torch.bfloat16  # 模型使用bfloat16精度以节省内存（新的配置项）
}

# 使用LLAMA2模型配置创建一个Llama2Model实例（7B版本）
model = Llama2Model(LLAMA2_CONFIG_7B)

# 计算模型的总参数数量
total_params = sum(p.numel() for p in model.parameters())  # 计算所有参数的元素总数
print(f"Total number of parameters: {total_params:,}")  # 打印参数总数，使用逗号分隔

# 定义一个函数，用于计算模型占用的内存大小（单位为GB）
def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0  # 初始化总参数量
    total_grads = 0   # 初始化总梯度量
    for param in model.parameters():
        # 计算每个参数的元素数量
        param_size = param.numel()  # 获取参数的元素总数
        total_params += param_size  # 累加参数数量
        # 检查该参数是否需要计算梯度（训练过程中需要梯度的参数）
        if param.requires_grad:
            total_grads += param_size  # 如果需要梯度，则累加梯度的元素数量

    # 计算模型的缓冲区大小（非参数的内存占用，如缓存的激活值）
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # 计算模型所占用的内存大小（单位：字节）
    element_size = torch.tensor(0, dtype=input_dtype).element_size()  # 获取每个元素的字节数
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size  # 总内存大小（字节数）

    # 将内存大小从字节转换为GB
    total_memory_gb = total_memory_bytes / (1024**3)  # 1GB = 1024^3字节

    return total_memory_gb  # 返回内存大小，单位GB

# 打印模型在float32精度下的内存占用
print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")

# 打印模型在bfloat16精度下的内存占用
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# 检查是否有可用的CUDA设备（GPU）
if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有CUDA设备，使用GPU
# 检查是否有可用的MPS设备（Apple设备的Metal Performance Shaders）
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # 如果有MPS设备，使用Apple的MPS
else:
    device = torch.device("cpu")  # 如果没有GPU或MPS，则使用CPU

# 将模型加载到相应的设备（GPU、MPS或CPU）
model.to(device)

# 导入 Hugging Face Hub 的登录功能，允许通过 API 进行身份验证
from huggingface_hub import login
import json  # 导入 JSON 模块，用于处理配置文件

# 打开并读取本地的 config.json 配置文件
with open("config.json", "r") as config_file:
    config = json.load(config_file)  # 解析 JSON 格式的配置文件
    access_token = config["HF_ACCESS_TOKEN"]  # 从配置中提取 Hugging Face 访问令牌

# 使用提取的令牌登录 Hugging Face Hub
login(token=access_token)

# 导入 Hugging Face Hub 的文件下载功能
from huggingface_hub import hf_hub_download

# 从 Hugging Face Hub 下载指定的模型文件（"tokenizer.model"）
tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",  # 模型的仓库 ID
    filename="tokenizer.model",  # 需要下载的文件名
    local_dir="Llama-2-7b"  # 下载后的文件存放路径
)

# 导入 SentencePiece 模块，用于处理模型的分词器
import sentencepiece as spm

# 定义一个 LlamaTokenizer 类，用于加载和使用 Llama 模型的分词器
class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()  # 创建 SentencePieceProcessor 实例
        sp.load(tokenizer_file)  # 加载分词器文件
        self.tokenizer = sp  # 保存加载的分词器

    # 定义一个 encode 方法，用于将文本编码为 token ID（数字化表示）
    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)  # 使用分词器将文本转换为 token IDs

    # 定义一个 decode 方法，用于将 token IDs 解码为原始文本
    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)  # 使用分词器将 token IDs 转换回文本

# 创建一个 LlamaTokenizer 实例，加载之前下载的分词器文件
tokenizer = LlamaTokenizer(tokenizer_file)

# 导入自定义的生成文本函数以及文本与 token IDs 相互转换的工具函数
from previous_chapters import generate, text_to_token_ids, token_ids_to_text

# 设置随机种子，保证结果可复现
torch.manual_seed(123)

# 调用自定义的生成函数，生成基于给定文本的模型输出
token_ids = generate(
    model=model,  # 指定要使用的模型（在此示例中未定义，需要在其他地方加载）
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),  # 将输入文本转换为 token IDs
    max_new_tokens=30,  # 限制生成的最大 token 数量为 30
    context_size=LLAMA2_CONFIG_7B["context_length"],  # 上下文窗口大小，来自模型配置
    top_k=1,  # 采样时只选择概率最高的前 1 个候选项（即贪心搜索）
    temperature=0.  # 生成时的温度，0 表示完全贪心选择
)

# 打印模型生成的文本结果
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  # 将 token IDs 解码为文本并打印出来


# 从Hugging Face Hub下载模型权重文件
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b",  # 指定模型的仓库ID
   filename="consolidated.00.pth",   # 指定下载的文件名
   local_dir="Llama-2-7b"            # 指定本地存储目录
)

# 加载下载的权重文件
weights = torch.load(weights_file, weights_only=True)

# 输出权重字典的前15个键（以检查权重文件的结构）
list(weights.keys())[:15]

# 定义一个函数用于检查并分配权重
def assign(left, right):
    # 如果左右张量的形状不一致，抛出异常
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    # 如果right是一个torch.Tensor类型，则克隆并返回一个新的torch.nn.Parameter
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        # 如果right不是Tensor，则将其转为Tensor并返回
        return torch.nn.Parameter(torch.tensor(right))

# 定义一个函数，将加载的权重赋值到Llama模型中
def load_weights_into_llama(model, param_config, params):
    # 加载词嵌入层的权重
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    # 遍历每一层
    for l in range(param_config["n_layers"]):

        # 加载自注意力层的权重
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"layers.{l}.attention.wq.weight"]  # 获取对应的权重
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"layers.{l}.attention.wk.weight"]
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # 加载前馈神经网络（FeedForward）的权重
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # 注意：w2和w3的顺序在权重文件中被颠倒，因此需要进行交换
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )

        # Load output layer weights
        model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
        model.out_head.weight = assign(model.out_head.weight, params["output.weight"])


load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights) # 加载权重
model.to(device) # 转移

# 设置随机种子，以确保可复现性
torch.manual_seed(123)

# 生成token_ids，通过模型生成新的文本
token_ids = generate(
    model=model,  # 使用的模型
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将输入文本转化为token ID
    max_new_tokens=25,  # 最大生成的新token数量
    context_size=LLAMA2_CONFIG_7B["context_length"],  # 上下文大小
    top_k=1,  # 在生成过程中选择最可能的下一个token
    temperature=0.  # 温度为0时意味着选择最高概率的token
)

# 打印生成的文本输出
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))





# 删除模型对象以释放内存
del model

# 使用 Hugging Face Hub 下载 Llama-2 7B 模型的权重文件
weights_file = hf_hub_download(
   repo_id="meta-llama/Llama-2-7b-chat",  # 模型仓库 ID
   filename="consolidated.00.pth",        # 权重文件名
   local_dir="Llama-2-7b-chat"            # 本地存储路径
)

# 初始化 Llama2Model 模型对象，并传入配置参数（这里是 7B 配置）
model = Llama2Model(LLAMA2_CONFIG_7B)

# 加载模型权重到模型中
load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)

# 将模型移到指定的设备（如 GPU 或 CPU）
model.to(device)

# 设置随机种子，以确保实验结果的可重现性
torch.manual_seed(123)

# 生成模型的输入 token 序列，这里是将文本 "What do llamas eat?" 转换为 token ID
token_ids = generate(
    model=model,                               # 要生成文本的模型
    idx=text_to_token_ids("What do llamas eat?", tokenizer).to(device),  # 将文本转换为 token IDs，并移动到设备
    max_new_tokens=25,                         # 生成的最大 token 数量
    context_size=LLAMA2_CONFIG_7B["context_length"],  # 上下文长度（即输入的最大 token 数量）
    top_k=1,                                   # Top-k 采样（选择概率最高的 k 个 token）
    temperature=0.                              # 温度控制采样的随机性，0 表示完全确定性采样
)

# 输出生成的文本，将生成的 token IDs 转换回文本形式
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))