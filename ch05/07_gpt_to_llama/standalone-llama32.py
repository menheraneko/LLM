import torch
import torch.nn as nn



#  1
# 定义一个前馈神经网络类（FeedForward），继承自nn.Module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义第一个线性层，输入维度为cfg["emb_dim"]，输出维度为cfg["hidden_dim"]，没有偏置项
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 定义第二个线性层，输入维度为cfg["emb_dim"]，输出维度为cfg["hidden_dim"]，没有偏置项
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 定义第三个线性层，输入维度为cfg["hidden_dim"]，输出维度为cfg["emb_dim"]，没有偏置项
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        # 通过fc1进行前向传播
        x_fc1 = self.fc1(x)
        # 通过fc2进行前向传播
        x_fc2 = self.fc2(x)
        # 使用SILU激活函数对fc1的输出进行激活，并与fc2的输出进行逐元素相乘
        x = nn.functional.silu(x_fc1) * x_fc2
        # 通过fc3进行最终的前向传播
        return self.fc3(x)

# 预计算ROPE（旋转位置编码）参数的函数
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    # 确保head_dim是偶数
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # 如果freq_config不为空，进行频率调整
    if freq_config is not None:
        # 计算低频和高频的波长
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        # 计算波长
        wavelen = 2 * torch.pi / inv_freq

        # 进行频率调整
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 平滑因子，用于频率的平滑调整
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        # 计算平滑后的逆频率
        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 标记中频的情况
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # 生成位置索引
    positions = torch.arange(context_length)

    # 计算角度
    angles = positions[:, None] * inv_freq[None, :]  # 形状: (context_length, head_dim // 2)

    # 扩展角度以匹配head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状: (context_length, head_dim)

    # 预计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


# 计算ROPE的函数
def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    # 确保head_dim是偶数
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 将x分为前半部分和后半部分
    x1 = x[..., : head_dim // 2]  # 第一部分
    x2 = x[..., head_dim // 2:]  # 第二部分

    # 用于ROPE计算的核心步骤（部分代码没有完成，假设是进行乘法等操作）

class SharedBuffers:
    _buffers = {}

    # 获取ROPE缓存的静态方法
    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        # 生成一个唯一的key来标识缓存
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        # 如果缓存中没有这个key，则创建新的缓存
        if key not in SharedBuffers._buffers:
            # 创建上三角矩阵mask
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            # 预计算ROPE的cos和sin
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            # 如果指定了dtype，则转换类型
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            # 将mask, cos, sin缓存到字典中
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]

# 定义一个分组查询注意力层（GroupedQueryAttention）
class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,
            rope_base=10_000,
            rope_config=None,
            dtype=None
        ):
        super().__init__()
        # 确保输出维度 d_out 能被头数 num_heads 整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        # 确保头数 num_heads 能被键值组数 num_kv_groups 整除
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out  # 保存输出维度
        self.num_heads = num_heads  # 保存头数
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 定义键、值的线性变换
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups  # 保存键值组数
        self.group_size = num_heads // num_kv_groups  # 每个键值组的头数

        # 定义查询的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # 定义输出投影
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # 使用 SharedBuffers 获取缓存
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        # 注册 mask, cos, sin 为模型的缓冲区
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        # 前向传播，计算注意力（具体实现略）
        b, num_tokens, d_in = x.shape  # 获取输入的形状

        queries = self.W_query(x)  # 计算查询，形状: (b, num_tokens, d_out)
        keys = self.W_key(x)  # 计算键，形状: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # 计算值，形状: (b, num_tokens, num_kv_groups * head_dim)

        # 重塑查询、键和值的形状
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # 形状:

# 定义一个变换器块（TransformerBlock）
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化注意力层
        self.att =  GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
            dtype=cfg["dtype"]
        )
        # 初始化前馈层
        self.ff = FeedForward(cfg)
        # 初始化归一化层
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))   # 输入x到注意力层，返回x
        x = x + shortcut  # 将原始输入加回

        # 前馈块的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))  # 输入x到前馈层


class Llama3Model(nn.Module):
    def __init__(self, cfg):
        # 初始化 Llama3Model 类，cfg 是包含模型配置的字典
        super().__init__()  # 调用父类的初始化方法

        # 创建一个嵌入层，输入大小为词汇表大小，输出的嵌入维度为 emb_dim
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # 创建多个 TransformerBlock 组成的序列，总层数为 n_layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 创建最终的归一化层，使用 RMSNorm，以便在最后的表示中标准化
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

        # 创建一个线性层，将输出维度映射到词汇表大小，以便进行分类
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        # 正向传播方法，in_idx 是输入的 token 索引

        # 通过嵌入层转换 token 索引为嵌入向量
        tok_embeds = self.tok_emb(in_idx)

        # 将嵌入向量传递到 Transformer 块中
        x = tok_embeds
        x = self.trf_blocks(x)  # 通过所有的 TransformerBlock
        x = self.final_norm(x)  # 对 Transformer 的输出进行归一化

        # 通过线性层转换为 logits，使用 bfloat16 类型
        logits = self.out_head(x.to(torch.bfloat16))
        return logits  # 返回 logits，用于后续的损失计算或预测






# 2
# Llama 3.2 1B 配置
LLAMA32_CONFIG = {
    "vocab_size": 128_256,      # 词汇表大小
    "context_length": 131_072,  # 上下文长度（即模型可以考虑的最大文本长度）
    "emb_dim": 2048,            # 嵌入维度（词嵌入的向量长度）
    "n_heads": 32,              # 注意力头的数量
    "n_layers": 16,             # Transformer模型的层数
    "hidden_dim": 8192,         # FeedForward层中间层的维度
    "n_kv_groups": 8,           # 注意力机制中，Key和Value的分组数
    "rope_base": 500_000.0,     # RoPE（相对位置编码）的基础值
    "dtype": torch.bfloat16,    # 使用bfloat16低精度数据类型来节省内存
    "rope_freq": {              # RoPE的频率缩放因子
        "factor": 32.0,         # RoPE频率缩放因子的常数
        "low_freq_factor": 1.0, # RoPE低频缩放因子
        "high_freq_factor": 4.0, # RoPE高频缩放因子
        "original_context_length": 8192, # 原始上下文长度
    }
}

# Llama 3.2 3B 配置（注释掉的部分，适用于不同规模的模型）
# LLAMA32_CONFIG = {
#     "vocab_size": 128_256,      # 词汇表大小
#     "context_length": 131_072,  # 上下文长度
#     "emb_dim": 3072,            # 嵌入维度
#     "n_heads": 24,              # 注意力头的数量
#     "n_layers": 28,             # Transformer层数
#     "hidden_dim": 8192,         # FeedForward中间层维度
#     "n_kv_groups": 8,           # Key-Value分组数
#     "rope_base": 500_000.0,     # RoPE基础值
#     "dtype": torch.bfloat16,    # 数据类型
#     "rope_freq": {              # RoPE频率参数
#         "factor": 32.0,         # RoPE频率缩放因子
#         "low_freq_factor": 1.0, # RoPE低频缩放因子
#         "high_freq_factor": 4.0, # RoPE高频缩放因子
#         "original_context_length": 8192, # 原始上下文长度
#     }
# }

# 判断使用的是1B还是3B模型，并设置相应的LLAMA_SIZE_STR
LLAMA_SIZE_STR = "1B" if LLAMA32_CONFIG["emb_dim"] == 2048 else "3B"

# 保存原来的上下文长度
old_context_length = LLAMA32_CONFIG["context_length"]
# 设置新的上下文长度为8192
LLAMA32_CONFIG["context_length"] = 8192

# 定义一个函数，用于重新缩放RoPE的theta值
def rescale_theta(theta_old, context_length_old, context_length_new):
    # 计算新的缩放因子
    scaling_factor = context_length_new / context_length_old
    # 计算新的theta值
    theta_new = theta_old * scaling_factor
    return theta_new

# 重新计算RoPE的theta，并更新配置中的rope_base
LLAMA32_CONFIG["rope_base"] = rescale_theta(
    LLAMA32_CONFIG["rope_base"],
    old_context_length,
    LLAMA32_CONFIG["context_length"]
)

# 打印新的RoPE theta值
print("New RoPE theta:", LLAMA32_CONFIG["rope_base"])

# 创建Llama3模型实例，并传入配置
model = Llama3Model(LLAMA32_CONFIG)

# 检查模型中某些张量的共享状态（是否相同）
print(model.trf_blocks[0].att.mask is model.trf_blocks[-1].att.mask)  # 检查第一个和最后一个Transformer块的mask是否共享
print(model.trf_blocks[0].att.cos is model.trf_blocks[-1].att.cos)    # 检查cosine张量是否共享
print(model.trf_blocks[0].att.sin is model.trf_blocks[-1].att.sin)    # 检查sin张量是否共享

# 计算并打印模型的总参数数量
total_params = sum(p.numel() for p in model.parameters())  # numel()返回张量中的元素数量
print(f"Total number of parameters: {total_params:,}")

# 计算去除共享词嵌入后的独立参数数量
total_params_normalized = total_params - model.tok_emb.weight.numel()  # 去除词嵌入的参数数量
print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

# 定义一个计算模型内存使用量的函数
def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    # 统计模型中所有参数的数量
    for param in model.parameters():
        param_size = param.numel()  # 每个参数的元素数量
        total_params += param_size
        if param.requires_grad:  # 如果参数需要计算梯度
            total_grads += param_size  # 统计需要梯度的参数的数量

    # 计算模型中所有缓冲区（非参数但需要占用内存的张量）的数量
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # 计算内存大小，单位为字节
    element_size = torch.tensor(0, dtype=input_dtype).element_size()  # 获取每个元素的字节大小
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size  # 总内存占用

    # 将字节转换为GB
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

# 打印模型的内存使用情况，分别使用float32和bfloat16数据类型
print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# 检查是否有可用的GPU或者MPS设备
if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有可用GPU，使用GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # 如果是MacOS并支持MPS，使用MPS
else:
    device = torch.device("cpu")   # 否则使用CPU

# 将模型加载到选定的设备上（GPU、MPS或CPU）
model.to(device)




# 3
import os  # 导入os模块，提供与操作系统交互的功能，例如文件操作
from pathlib import Path  # 导入Path类，用于文件路径操作

import tiktoken  # 导入tiktoken库，用于处理tokenization（分词）
from tiktoken.load import load_tiktoken_bpe  # 从tiktoken库中导入load_tiktoken_bpe方法，用于加载BPE（Byte Pair Encoding）模型


class Tokenizer:
    def __init__(self, model_path):
        # 检查给定的模型文件路径是否存在
        assert os.path.isfile(model_path), f"Model file {model_path} not found"

        # 加载合并的 BPE 排序
        mergeable_ranks = load_tiktoken_bpe(model_path)

        # 定义特殊标记的字典
        self.special_tokens = {
            "<|begin_of_text|>": 128000,  # 文本开始的标记
            "<|end_of_text|>": 128001,  # 文本结束的标记
            "<|start_header_id|>": 128006,  # 头部开始标记
            "<|end_header_id|>": 128007,  # 头部结束标记
            "<|eot_id|>": 128009,  # 结束标记 ID
        }
        # 为 256 个保留标记 ID 创建唯一的标记
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        # 创建一个编码对象，支持特定的正则表达式和合并规则
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,  # 使用模型文件的名称
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,  # 将合并规则传递给编码器
            special_tokens=self.special_tokens  # 向编码器传递特殊标记
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        # 如果指定了开始标记，则将其添加到 tokens 列表中
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

            # 对文本进行编码，并加入可能的特殊标记
        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        # 如果指定了结束标记，则将其添加
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        # 解码 token 列表并返回对应文本
        return self.model.decode(tokens)


class ChatFormat:
    def __init__(self, tokenizer):
        # 初始化 ChatFormat，并保存 tokenizer 实例
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        # 添加头部开始标记
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        # 对消息角色进行编码并添加到 tokens 列表中
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        # 添加头部结束标记
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        # 为换行符添加额外的 token
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode(self, text):
        # 将消息构造成字典格式
        message = {
            "role": "user",  # 消息角色设为用户
            "content": text  # 消息内容
        }

        # 编码消息头
        tokens = self.encode_header(message)
        # 编码用户内容，并添加到 token 列表中
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        # 添加结束 token
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def decode(self, token_ids):
        # 解码 token ID 列表并返回对应文本
        return self.tokenizer.decode(token_ids)

# 导入huggingface_hub库，并执行登录操作
from huggingface_hub import login

login()# 执行huggingface的登录操作，允许访问私有模型和数据

# 从huggingface_hub库导入hf_hub_download函数，用于下载模型文件
from huggingface_hub import hf_hub_download

# 使用hf_hub_download函数下载指定模型的tokenizer文件
tokenizer_file_path = hf_hub_download(
      repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",# 模型仓库ID，LLAMA_SIZE_STR应该是某个变量
      filename="original/tokenizer.model",   # 下载的文件路径（模型的tokenizer文件）
      local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"# 将文件保存到本地的指定目录
)

# 初始化Tokenizer类，并传入下载的模型文件路径
tokenizer = Tokenizer(tokenizer_file_path)

# 使用初始化的tokenizer实例来创建ChatFormat类
chat_tokenizer = ChatFormat(tokenizer)







# 4
# 定义 assign 函数，用于将权重从右侧张量复制到左侧张量
def assign(left, right, tensor_name="unknown"):
    # 检查两个张量的形状是否一致，如果不一致则抛出错误
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    # 如果右侧张量是 torch.Tensor 类型，克隆并分离它，返回 Parameter 类型的张量
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        # 如果右侧不是 tensor，则先转为 tensor 然后返回
        return torch.nn.Parameter(torch.tensor(right))

# 定义函数 load_weights_into_llama，用于将加载的权重赋值到 Llama 模型
def load_weights_into_llama(model, param_config, params):
    # 为模型的 tok_emb 权重赋值
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    # 遍历每一层，加载其相应的权重
    for l in range(param_config["n_layers"]):

        # 加载 Attention 层的权重
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # 加载 FeedForward 层的权重
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )

# 从 safetensors.torch 模块导入 load_file 函数，用于加载 safetensors 格式的权重文件
from safetensors.torch import load_file

# 判断 Llama 模型的大小，并根据大小加载不同的权重文件
if LLAMA_SIZE_STR == "1B":
    # 如果是 1B 模型，则直接下载并加载单一权重文件
    weights_file = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",  # 预训练模型的 repo 名称
        filename=f"model.safetensors",  # 权重文件名
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"  # 下载后保存的本地路径
    )
    combined_weights = load_file(weights_file)  # 加载 safetensors 格式的权重文件

else:
    # 如果不是 1B 模型，则可能有多个分片文件，加载并合并它们
    combined_weights = {}
    for i in range(1, 3):  # 假设有两个权重文件（编号从 1 到 2）
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",  # 预训练模型的 repo 名称
            filename=f"model-0000{i}-of-00002.safetensors",  # 分片文件名
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"  # 下载后保存的本地路径
        )
        current_weights = load_file(weights_file)  # 加载当前的分片权重文件
        combined_weights.update(current_weights)  # 将当前的权重更新到 combined_weights 中

# 将加载的权重赋值到模型中
load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)

# 将模型移动到指定的设备（例如 GPU）
model.to(device)

# 删除 combined_weights，释放内存
del combined_weights  # free up memory

# 打印权重绑定信息，检查 token embedding 和 output head 的权重是否相同（权重绑定检查）
print("Weight tying:", torch.equal(model.tok_emb.weight, model.out_head.weight))







# 5
def text_to_token_ids(text, tokenizer):
    # 使用 tokenizer 对输入文本进行编码，返回 token 的 ID 列表
    encoded = tokenizer.encode(text)
    # 将编码后的 ID 列表转换为 PyTorch tensor，并增加一个批量维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # 去掉批量维度（如果有的话）
    flat = token_ids.squeeze(0)  # remove batch dimension
    # 使用 tokenizer 将 token ID 列表解码成文本
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 生成文本的过程：根据输入的 token ID 和条件生成新的 token
    for _ in range(max_new_tokens):
        # 只保留输入的最后 context_size 个 token 作为上下文
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():  # 禁用梯度计算（推理时不需要梯度）
            logits = model(idx_cond)  # 获取模型输出的 logits
        logits = logits[:, -1, :]  # 只保留最后一个时间步的 logits

        # 如果指定了 top_k，进行 top-k 采样
        if top_k is not None:
            # 获取 top_k 的最大 logits 值
            top_logits, _ = torch.topk(logits, top_k)
            # 获取 logits 中最小的 top_k 值
            min_val = top_logits[:, -1]
            # 将小于 top_k 最小值的 logits 设置为负无穷，避免被采样
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果指定了 temperature，进行温度调节
        if temperature > 0.0:
            # 对 logits 进行温度缩放
            logits = logits / temperature

            # 通过 softmax 将 logits 转换为概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 从概率分布中采样得到下一个 token 的索引
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则，直接选择 logits 最大的索引作为下一个 token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果生成了 end-of-sequence（eos）token，并且指定了 eos_id，则提前停止生成
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # 将生成的 token 索引追加到当前的 token 序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


# 设置输入文本（Prompt）
PROMPT = "What do llamas eat?"

# 设置随机种子，确保结果可复现
torch.manual_seed(123)

# 调用生成函数生成 token 序列
token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),  # 将 prompt 转换为 token ID，移动到正确的设备
    max_new_tokens=150,  # 最多生成 150 个新 token
    context_size=LLAMA32_CONFIG["context_length"],  # 上下文窗口大小，来自模型配置
    top_k=1,  # 设置 top_k 为 1，表示只采样最可能的 token
    temperature=0.  # 设置温度为 0，表示选择概率最大的 token
)

# 将生成的 token ID 转换为文本
output_text = token_ids_to_text(token_ids, tokenizer)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # 在文本中找到指定的标记 header_end 的位置
    index = text.find(header_end)

    if index != -1:
        # 如果找到了标记，则返回标记之后的文本，去掉多余的空白字符
        return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
    else:
        # 如果没有找到标记，返回原文本
        return text


# 输出清理后的文本
print("Output text:\n", clean_text(output_text))