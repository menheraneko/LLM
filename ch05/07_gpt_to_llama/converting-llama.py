
import os  # 导入 os 模块，用于操作文件和目录路径
import sys  # 导入 sys 模块，用于访问 Python 运行时的相关参数
import io  # 导入 io 模块，用于处理文件输入输出
import nbformat  # 导入 nbformat 模块，用于读取 Jupyter Notebook 文件
import types  # 导入 types 模块，用于创建新的 Python 模块类型



# 1.1
# 定义一个从 Jupyter Notebook 导入定义的函数
def import_from_notebook():
    # 定义一个辅助函数，用于从 Notebook 中导入特定的函数或类定义
    def import_definitions_from_notebook(fullname, names):
        # 获取当前工作目录路径
        current_dir = os.getcwd()
        # 构造 Notebook 文件的完整路径，假设文件名为 fullname.ipynb
        path = os.path.join(current_dir, fullname + ".ipynb")
        # 将路径标准化，处理路径中的符号（如 . 或 ..）
        path = os.path.normpath(path)

        # 加载 Notebook 文件
        if not os.path.exists(path):
            raise FileNotFoundError(f"Notebook file not found at: {path}")  # 如果文件不存在，抛出异常

        # 打开 Notebook 文件，读取其内容
        with io.open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)  # 读取文件并解析为 nbformat 格式

        # 创建一个新的模块，用来存储从 Notebook 导入的函数和类
        mod = types.ModuleType(fullname)
        sys.modules[fullname] = mod  # 将新模块加入到 sys.modules 中

        # 遍历 Notebook 中的每个单元格，查找其中的函数或类定义
        for cell in nb.cells:
            if cell.cell_type == "code":  # 只处理代码单元格
                cell_code = cell.source  # 获取单元格中的源代码
                for name in names:
                    # 检查代码中是否包含函数或类的定义（例如：def precompute_rope_params 或 class FeedForward）
                    if f"def {name}" in cell_code or f"class {name}" in cell_code:
                        exec(cell_code, mod.__dict__)  # 执行代码并将定义添加到模块中

        return mod  # 返回创建的模块

    # 设置需要导入的 Notebook 文件名和函数/类名
    fullname = "converting-gpt-to-llama2"  # 指定 Notebook 文件名（不带扩展名）
    names = ["precompute_rope_params", "compute_rope", "SiLU", "FeedForward", "RMSNorm",
             "MultiHeadAttention"]  # 指定需要导入的函数和类名称

    return import_definitions_from_notebook(fullname, names)  # 调用函数并返回模块


# 调用 import_from_notebook 函数导入 Notebook 中的内容
imported_module = import_from_notebook()

# 如果需要，可以重新定义 `precompute_rope_params`，但此行代码被注释掉了
# precompute_rope_params = getattr(imported_module, "precompute_rope_params", None)

# 使用 `getattr` 从导入的模块中获取函数或类的定义，如果没有找到则返回 None
compute_rope = getattr(imported_module, "compute_rope", None)
SiLU = getattr(imported_module, "SiLU", None)
FeedForward = getattr(imported_module, "FeedForward", None)
RMSNorm = getattr(imported_module, "RMSNorm", None)

# 仅用于比较 purposes 获取 `MultiHeadAttention` 的定义
MultiHeadAttention = getattr(imported_module, "MultiHeadAttention", None)




# 1.2
import torch


# 定义预计算 RoPE（旋转位置编码）参数的函数
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    # 确保 head_dim 为偶数，因为 RoPE 的计算需要
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    ################################ NEW ###############################################
    # 频率调整部分
    if freq_config is not None:
        # 从配置中获取低频和高频的波长调整因子
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        # 计算每个频率对应的波长
        wavelen = 2 * torch.pi / inv_freq

        # 调整频率，根据条件修改 inv_freq（例如，调整低频部分）
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 计算平滑因子，用于频率的平滑调整
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
                freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        # 应用平滑因子，调整频率
        smoothed_inv_freq = (
                (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 判断波长是否处于中频范围内
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        # 更新最终的 inv_freq
        inv_freq = inv_freq_llama
    ####################################################################################

    # 生成位置索引（即位置编码的索引）
    positions = torch.arange(context_length)

    # 计算角度：每个位置与每个频率的乘积
    angles = positions[:, None] * inv_freq[None, :]  # 结果形状: (context_length, head_dim // 2)

    # 扩展角度矩阵以匹配 head_dim
    angles = torch.cat([angles, angles], dim=1)  # 结果形状: (context_length, head_dim)

    # 预计算 sin 和 cos 值
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 返回 cos 和 sin 值，这些值将用于后续的 RoPE 编码计算
    return cos, sin


# 设置 Llama 模型的上下文长度和 theta_base
llama_2_context_len = 4096
llama_3_context_len = 8192

llama_2_theta_base = 10_000
llama_3_theta_base = 500_000

# 设置批量大小、头数和每个头的维度
batch_size = 2
num_heads = 4
head_dim = 16

# 实例化 RoPE 参数，使用 Llama 3 模型的参数
cos, sin = precompute_rope_params(
    head_dim=head_dim,  # 每个头的维度
    theta_base=llama_3_theta_base,  # Llama 3 模型的 theta_base
    context_length=llama_3_context_len  # Llama 3 模型的上下文长度
)

# 生成随机的查询（queries）和键（keys）张量，作为示例数据
torch.manual_seed(123)  # 设置随机种子，保证结果可复现
queries = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)
keys = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)

# 将查询和键应用 RoPE，进行位置编码（Rotary Position Embedding）
queries_rot = compute_rope(queries, cos, sin)  # 对查询进行 RoPE 编码
keys_rot = compute_rope(keys, cos, sin)  # 对键进行 RoPE 编码

# 1.3
import torch.nn as nn


############################# NEW  #############################
# 定义一个用于缓存共享的类
class SharedBuffers:
    # 使用字典缓存每个共享的缓冲区
    _buffers = {}

    # 静态方法获取或创建共享的缓存
    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        # 根据输入参数生成唯一的键值来标识缓存
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        # 如果缓存中没有对应的键，则创建新缓存
        if key not in SharedBuffers._buffers:
            # 创建一个上三角矩阵，表示mask
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            # 预计算位置编码的cos和sin值
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            # 如果指定了数据类型，将cos和sin转换为相应的数据类型
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            # 将计算得到的mask、cos和sin缓存在字典中
            SharedBuffers._buffers[key] = (mask, cos, sin)

        # 返回缓存的结果
        return SharedBuffers._buffers[key]


############################# NEW  #############################

# 定义一个多头注意力（MHA）模块的变体，使用了分组查询的注意力机制
class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,  # 新增参数：KV分组的数量
            rope_base=10_000,  # 新增参数：位置编码的基数
            rope_config=None,  # 新增参数：位置编码的配置
            dtype=None  # 新增参数：数据类型
    ):
        super().__init__()
        # 检查输出维度是否可以被头数整除
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"
        # 检查头数是否可以被KV分组数整除
        assert num_heads % num_kv_groups == 0, "num_heads必须能被num_kv_groups整除"

        # 设置类的属性
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        ############################# NEW  #############################
        # 定义key和value的线性变换，将输入d_in映射到分组的head_dim维度
        # W_key和W_value的权重矩阵大小为(d_in, num_kv_groups * head_dim)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups  # KV分组数量
        self.group_size = num_heads // num_kv_groups  # 每个KV组的头数
        ############################################################


# 设置一些参数
batch_size = 1  # 批大小
context_len = 3000  # 当前上下文长度
max_context_len = 8192  # 最大上下文长度
embed_dim = 4096  # 嵌入维度
num_heads = 32  # 头数

# 创建一个示例输入张量，形状为(batch_size, context_len, embed_dim)
example_batch = torch.randn((batch_size, context_len, embed_dim))

# 创建一个多头注意力模型（MHA）
mha = MultiHeadAttention(
    d_in=embed_dim,  # 输入的维度
    d_out=embed_dim,  # 输出的维度
    context_length=max_context_len,  # 上下文长度
    num_heads=num_heads  # 头数
)

# 执行模型的前向传播
mha(example_batch)

# 打印W_key、W_value和W_query的权重矩阵形状
print("W_key:", mha.W_key.weight.shape)
print("W_value:", mha.W_value.weight.shape)
print("W_query:", mha.W_query.weight.shape)

# 创建一个分组查询注意力模型（GQA）
gqa = GroupedQueryAttention(
    d_in=embed_dim,  # 输入的维度
    d_out=embed_dim,  # 输出的维度
    context_length=max_context_len,  # 上下文长度
    num_heads=num_heads,  # 头数
    num_kv_groups=8,  # KV分组数
    rope_base=llama_3_theta_base  # 位置编码的基数（假设llama_3_theta_base已定义）
)

# 执行分组查询注意力的前向传播
gqa(example_batch)

# 打印分组查询注意力模型的权重矩阵形状
print("W_key:", gqa.W_key.weight.shape)
print("W_value:", gqa.W_value.weight.shape)
print("W_query:", gqa.W_query.weight.shape)

# 打印模型的总参数量
print("Total number of parameters:")

# 计算多头注意力模型的总参数量
mha_total_params = sum(p.numel() for p in mha.parameters())
print(f"MHA: {mha_total_params:,}")

# 计算分组查询注意力模型的总参数量
gqa_total_params = sum(p.numel() for p in gqa.parameters())
print(f"GQA: {gqa_total_params:,}")

# 释放内存
del mha
del gqa




# 1.4
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 初始化父类 nn.Module
        # 定义多头注意力层（这里用的是 GroupedQueryAttention, 类似 MultiHeadAttention）
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],  # 输入的特征维度，即embedding的维度
            d_out=cfg["emb_dim"],  # 输出的特征维度，通常与输入相同
            context_length=cfg["context_length"],  # 上下文长度，指序列的最大长度
            num_heads=cfg["n_heads"],  # 注意力头的数量
            num_kv_groups=cfg["n_kv_groups"],  # KV（Key-Value）对的分组数量（新的参数）
            rope_base=cfg["rope_base"],  # 相对位置编码的基数（新的参数）
            rope_config=cfg["rope_freq"],  # 相对位置编码的频率配置（新的参数）
            dtype=cfg["dtype"]  # 数据类型（如torch.float32等）
        )
        # 定义前馈神经网络层
        self.ff = FeedForward(cfg)
        # 定义第一层RMSNorm归一化层，通常用于归一化层的稳定性
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        # 定义第二层RMSNorm归一化层
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x):
        # 残差连接：保存输入x，用于后续的加法操作
        shortcut = x
        # 对输入进行归一化
        x = self.norm1(x)
        # 使用注意力机制处理输入，注意力输出的形状是 [batch_size, num_tokens, emb_size]
        x = self.att(x.to(torch.bfloat16))  # 将输入转换为bfloat16类型，以减少计算资源
        # 将原始输入加回（残差连接），使得信息流能够保持
        x = x + shortcut  # Add the original input back

        # 另一层残差连接：保存当前的输入x，用于后续的加法操作
        shortcut = x
        # 对x进行第二次归一化
        x = self.norm2(x)
        # 通过前馈神经网络层进行处理
        x = self.ff(x.to(torch.bfloat16))  # 同样将输入转换为bfloat16类型
        # 将原始输入加回（残差连接），保持梯度流通
        x = x + shortcut  # Add the original input back

        # 返回经过处理后的输出
        return x






# 1.5
# 定义Llama3模型，继承自nn.Module类
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        # 调用父类（nn.Module）的构造函数
        super().__init__()

        # 初始化词嵌入层，将词汇表大小映射到嵌入维度
        # cfg["vocab_size"]是词汇表的大小，cfg["emb_dim"]是嵌入的维度，cfg["dtype"]是数据类型
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # 创建多个Transformer块，数量为cfg["n_layers"]
        # TransformerBlock是一个自定义的类，用来构建每一层的Transformer模块
        self.trf_blocks = nn.Sequential(
            # 使用列表推导生成多个Transformer块（层）
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 最后的归一化层，使用RMSNorm来对输入进行归一化处理
        # cfg["emb_dim"]是输入的维度大小，eps=1e-5是为了防止除以零的数值稳定性
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)

        # 输出头，将模型的嵌入空间映射到词汇表大小，用于生成预测的词汇
        # cfg["emb_dim"]是输入维度，cfg["vocab_size"]是输出词汇的数量
        # bias=False表示不使用偏置项
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    # 定义模型的前向传播函数
    def forward(self, in_idx):
        # 获取输入索引的词嵌入表示
        tok_embeds = self.tok_emb(in_idx)

        # 将词嵌入传入Transformer块中进行处理
        x = tok_embeds
        x = self.trf_blocks(x)

        # 对Transformer输出进行归一化处理
        x = self.final_norm(x)

        # 使用线性层将模型的输出映射到词汇表大小的logits
        # logits是对每个位置的词汇预测值
        logits = self.out_head(x.to(torch.bfloat16))

        # 返回模型的输出logits
        return logits


# 定义 Llama2 的配置参数字典
LLAMA2_CONFIG_7B = {
    "vocab_size": 32_000,  # 词汇表的大小
    "context_length": 4096,  # 上下文窗口的长度
    "emb_dim": 4096,  # 嵌入层的维度
    "n_heads": 32,  # 注意力头的数量
    "n_layers": 32,  # Transformer 层数
    "hidden_dim": 11_008,  # FeedForward 中间层的维度
    "dtype": torch.bfloat16  # 使用较低精度的 bfloat16 数据类型来节省内存
}

# 定义 Llama3 的配置参数字典，参数有更新
LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,  # 更新：更大的词汇表大小
    "context_length": 8192,  # 更新：更大的上下文窗口长度
    "emb_dim": 4096,  # 嵌入层的维度
    "n_heads": 32,  # 注意力头的数量
    "n_layers": 32,  # Transformer 层数
    "hidden_dim": 14_336,  # 更新：更大的 FeedForward 中间层的维度
    "n_kv_groups": 8,  # 更新：用于分组查询注意力的键值对组数
    "rope_base": 500_000.0,  # 更新：RoPE（旋转位置编码）中的 base 值增大
    "rope_freq": None,  # 更新：调整 RoPE 频率的额外配置
    "dtype": torch.bfloat16  # 使用较低精度的 bfloat16 数据类型来节省内存
}

# 初始化 Llama3 模型，使用上述配置参数
model = Llama3Model(LLAMA3_CONFIG_8B)

# 检查模型的不同 Transformer 层中注意力掩码和旋转编码是否相同
# 比较模型第一层和最后一层的注意力掩码是否相同
print(model.trf_blocks[0].att.mask is model.trf_blocks[-1].att.mask)
# 比较模型第一层和最后一层的余弦值（cos）是否相同
print(model.trf_blocks[0].att.cos is model.trf_blocks[-1].att.cos)
# 比较模型第一层和最后一层的正弦值（sin）是否相同
print(model.trf_blocks[0].att.sin is model.trf_blocks[-1].att.sin)

# 计算模型的总参数量
total_params = sum(p.numel() for p in model.parameters())  # 通过对所有参数的元素数量求和来计算总参数数
print(f"Total number of parameters: {total_params:,}")  # 打印出模型的总参数量，格式化为千分位显示


# 定义一个函数来计算模型的内存占用大小
def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0  # 用于累加总参数数量
    total_grads = 0  # 用于累加总梯度数量
    for param in model.parameters():
        # 计算每个参数的元素数量
        param_size = param.numel()
        total_params += param_size  # 累加参数的总元素数
        # 如果参数需要计算梯度，则累加梯度的元素数量
        if param.requires_grad:
            total_grads += param_size

    # 计算模型所有缓冲区（非参数）的内存大小
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # 计算总的内存大小（字节）= (元素个数) * (每个元素的字节大小)
    # 假设参数和梯度使用与输入数据类型相同的精度
    element_size = torch.tensor(0, dtype=input_dtype).element_size()  # 获取每个元素的大小（字节）
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size  # 计算总内存大小（字节）

    # 将字节转换为千兆字节（GB）
    total_memory_gb = total_memory_bytes / (1024 ** 3)  # 转换为 GB

    return total_memory_gb  # 返回计算出来的内存大小（GB）


# 打印模型在不同数据类型下的内存占用
print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

# 检测是否支持 GPU（CUDA），如果支持则使用 GPU 否则使用 CPU 或 MPS（MacOS上的Metal）
if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有 GPU，则使用 CUDA 设备
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # 如果是 MacOS 且支持 MPS，则使用 MPS 设备
else:
    device = torch.device("cpu")  # 如果没有 GPU 和 MPS，则使用 CPU

# 将模型移动到相应的设备（GPU 或 CPU）
model.to(device)




import os  # 导入操作系统相关模块，用于文件路径操作
from pathlib import Path  # 导入路径操作模块，用于处理路径

import tiktoken  # 导入tiktoken模块，处理文本分词
from tiktoken.load import load_tiktoken_bpe  # 从tiktoken库中导入加载bpe分词的方法


class Tokenizer:
    def __init__(self, model_path):
        # 初始化 Tokenizer 类，model_path 是模型文件的路径
        assert os.path.isfile(model_path), f"Model file {model_path} not found"  # 检查模型文件是否存在
        mergeable_ranks = load_tiktoken_bpe(model_path)  # 加载 BPE 合并规则

        # 定义特殊标记及其对应的 ID
        self.special_tokens = {
            "<|begin_of_text|>": 128000,  # 文本开始标记
            "<|end_of_text|>": 128001,  # 文本结束标记
            "<|start_header_id|>": 128006,  # header 开始标记
            "<|end_header_id|>": 128007,  # header 结束标记
            "<|eot_id|>": 128009,  # 结束标记
        }

        # 生成并更新保留标记，范围从 128002 到 128257
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        # 创建 tiktoken 编码实例，使用指定的模式和合并规则
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,  # 使用模型文件名作为名称
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            # 用于匹配的正则表达式
            mergeable_ranks=mergeable_ranks,  # BPE 合并规则
            special_tokens=self.special_tokens  # 特殊标记
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        # 编码输入文本，参数表示是否添加开始/结束标记以及允许或禁止的特殊标记
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]  # 如果需要，添加文本开始标记
        else:
            tokens = []

            # 使用模型进行编码，合并允许和禁止的特殊标记
        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])  # 如果需要，添加文本结束标记

        return tokens  # 返回编码后的 token 列表

    def decode(self, tokens):
        # 解码 token 列表回原文本
        return self.model.decode(tokens)  # 使用模型解码


# 导入Huggingface Hub的API，用于登录和下载模型
from huggingface_hub import login  # 导入Huggingface Hub的登录功能
import json  # 导入json模块，用于读取配置文件

# 读取配置文件，获取Huggingface的API访问令牌
with open("config.json", "r") as config_file:  # 打开配置文件
     config = json.load(config_file)  # 解析配置文件内容
     access_token = config["HF_ACCESS_TOKEN"]  # 提取API访问令牌

# 使用访问令牌登录Huggingface Hub
login(token=access_token)

# 导入Huggingface Hub的下载API
from huggingface_hub import hf_hub_download

# 下载指定模型的tokenizer文件
tokenizer_file_path = hf_hub_download(
     repo_id = "meta-llama/Meta-Llama-3-8B",  # 模型仓库ID
     filename = "original/tokenizer.model",  # tokenizer模型文件名
     local_dir = "Llama-3-8B"  # 下载到本地的目录
)



# 从上一章节导入生成、文本到token ID、token ID到文本的相关函数
from previous_chapters import generate, text_to_token_ids, token_ids_to_text

# 设置随机种子，确保实验的可重复性
torch.manual_seed(123)

tokenizer = Tokenizer(tokenizer_file_path)

# 生成token ID，基于"Every effort"这段文本
token_ids = generate(
     model = model,  # 使用的模型
     idx = text_to_token_ids("Every effort", tokenizer).to(device),  # 将文本转为token ID并迁移到设备
     max_new_tokens = 30,  # 最大生成的token数量
     context_size = LLAMA3_CONFIG_8B["context_length"],  # 上下文长度配置
     top_k = 1,  # 设置top-k采样策略，1表示只选择最可能的下一个token
     temperature = 0.  # 温度设置，0表示完全贪婪选择
)

# 输出生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 导入safetensors库中的load_file函数，用于加载safetensors格式的文件
from safetensors.torch import load_file

# 初始化一个空字典，用于存储合并后的权重
combined_weights = {}

# 循环加载每个权重文件（1到4）
for i in range(1, 5):
    # 下载指定的模型权重文件，并存储到本地目录" Llama-3-8B"
    weights_file = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",  # 模型的repo ID
        filename=f"model-0000{i}-of-00004.safetensors",  # 文件名，依次加载多个文件
        local_dir="Llama-3-8B"  # 本地存储目录
    )
    # 使用load_file函数加载safetensors格式的权重文件
    current_weights = load_file(weights_file)
    # 将当前加载的权重合并到combined_weights字典中
    combined_weights.update(current_weights)


# 定义一个函数assign，用于将权重赋值给模型参数
def assign(left, right, tensor_name="unknown"):
    # 如果左右两者的形状不匹配，抛出一个ValueError
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    # 如果右边是一个tensor，则返回它的副本作为torch.nn.Parameter
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        # 否则，将右边的值转换为tensor，并作为torch.nn.Parameter返回
        return torch.nn.Parameter(torch.tensor(right))


# 定义一个函数，加载权重到Llama模型中
def load_weights_into_llama(model, param_config, params):
    # 将"model.embed_tokens.weight"的权重加载到模型的tok_emb层
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"],
                                  "model.embed_tokens.weight")

    # 遍历每一层（层数由param_config["n_layers"]指定）
    for l in range(param_config["n_layers"]):
        # 加载每一层的Attention模块的权重
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],  # 从params中提取对应的权重
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

        # 加载每一层的FeedForward模块（MLP）的权重
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        # 注意代码在这里可能有遗漏（没有完整显示），应该还会加载其他FeedForward模块的权重


# 设置随机种子，以确保结果可重现
torch.manual_seed(123)

# 使用生成器生成文本，以下是一些参数的设置
token_ids = generate(
    model=model,  # 指定用于生成文本的模型
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将输入文本转化为token ID，并移至设备
    max_new_tokens=25,  # 生成的最大新词数量
    context_size=LLAMA3_CONFIG_8B["context_length"],  # 上下文窗口大小
    top_k=1,  # top_k sampling策略，选择概率最高的词
    temperature=0.0  # temperature控制生成的随机性，0.0意味着选择最有可能的词
)

# 打印生成的文本结果
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

5
# 引入垃圾回收模块
import gc

# 删除现有的模型对象，释放内存
del model

# 手动触发Python垃圾回收，清理无用的内存
gc.collect() # 运行Python垃圾回收器

# 如果CUDA可用，清空显存缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 创建一个空的字典，用来存放合并后的权重数据
combined_weights = {}

# 循环加载模型的各个权重文件
for i in range(1, 5):  # 假设共有4个权重文件，循环加载
    combined_weights = {}  # 初始化一个字典，用于存储合并后的权重

    # 内部循环加载每个权重文件
    for j in range(1, 5):  # 注意这里使用 j 作为循环变量，避免与外层循环的 i 冲突
        # 下载当前权重文件
        weights_file = hf_hub_download(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # 模型的 Hugging Face Hub 仓库 ID
            filename=f"model-0000{j}-of-00004.safetensors",  # 权重文件名，根据循环变量 j 生成
            local_dir="Llama-3-8B-Instruct"  # 本地保存路径
        )

        # 加载当前权重文件
        current_weights = load_file(weights_file)

        # 合并当前权重到 combined_weights 字典中
        combined_weights.update(current_weights)

# 初始化 Llama3 模型
model = Llama3Model(LLAMA3_CONFIG_8B)
# 将合并的权重加载到模型中
load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)
# 将模型移动到指定的设备（如 GPU）
model.to(device)
# 删除 combined_weights 字典以释放内存
del combined_weights
# 定义一个聊天格式类，包含对话内容的编码和解码方法
class ChatFormat:
    def __init__(self, tokenizer):
        # 初始化 ChatFormat 类，tokenizer 是用于编码和解码的 Tokenizer 实例
        self.tokenizer = tokenizer

    def encode_header(self, message):
        # 编码消息的标题部分
        tokens = []  # 初始化一个空列表用于存放 token

        # 添加 header 开始特殊标记
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])

        # 编码消息角色并添加到 token 列表
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))

        # 添加 header 结束特殊标记
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])

        # 编码两个换行符，并添加到 token 列表
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))

        return tokens  # 返回编码后的 token 列表

    def encode(self, text):
        # 编码用户消息内容
        message = {
            "role": "user",  # 消息角色设为 "user"
            "content": text  # 消息内容
        }

        # 编码消息的标题部分
        tokens = self.encode_header(message)

        # 编码内容并添加到 token 列表
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )

        # 添加结束标记
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])

        return tokens  # 返回完整的编码 token 列表

    def decode(self, token_ids):
        # 将 token ID 解码为文本
        return self.tokenizer.decode(token_ids)

    # 创建 ChatFormat 实例


chat_tokenizer = ChatFormat(tokenizer)

# 编码示例文本 "Hello World!"
token_ids = chat_tokenizer.encode("Hello World!")
print(token_ids)  # 输出编码后的 token IDs

# 解码 token IDs 回原文本
decoded_text = tokenizer.decode(token_ids)

torch.manual_seed(123)  # 设置随机种子，确保生成结果可复现

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("What do llamas eat?", chat_tokenizer).to(device),
    max_new_tokens=150,  # 生成的最大新 token 数量为 150
    context_size=LLAMA3_CONFIG_8B["context_length"],  # 上下文大小
    top_k=1,  # 采样时选择最高概率的 k 个候选（这里是 1）
    temperature=0.  # 温度设置为 0，表示选择确定性的输出
)

# 将生成的 token IDs 转换为文本
output_text = token_ids_to_text(token_ids, tokenizer)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # 定义一个函数用于清理文本，删除 header 部分
    # 查找第一个 "<|end_header_id|>" 的索引
    index = text.find(header_end)

    if index != -1:
        # 返回 header 结束标记后的子字符串，并去掉首尾空白
        return text[index + len(header_end):].strip()
    else:
        # 如果未找到标记，返回原文本
        return text

# 输出清理后的结果
print("Output text:\n", clean_text(output_text))
LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 8192,  # Context length
    "emb_dim": 4096,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 32,  # Number of layers
    "hidden_dim": 14_336,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "rope_freq": None,  # Additional configuration for adjusting the RoPE frequencies
    "dtype": torch.bfloat16  # Lower-precision dtype to save memory
}

LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # NEW: Larger supported context length
    "emb_dim": 4096,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 32,  # Number of layers
    "hidden_dim": 14_336,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
    "rope_freq": {  # NEW: RoPE frequency scaling
       "factor": 8.0,
       "low_freq_factor": 1.0,
       "high_freq_factor": 4.0,
       "original_context_length": 8192,
    }
}


# 保存旧的上下文长度
old_context_length = LLAMA31_CONFIG_8B["context_length"]
# 更新新的上下文长度
LLAMA31_CONFIG_8B["context_length"] = 8192

def rescale_theta(theta_old, context_length_old, context_length_new):
    # 根据新的上下文长度调整 RoPE 参数
    scaling_factor = context_length_new / context_length_old  # 计算缩放因子
    theta_new = theta_old * scaling_factor  # 按照缩放因子调整 theta
    return theta_new  # 返回新的 theta

# 使用 rescale_theta 函数更新 rope_base 参数
LLAMA31_CONFIG_8B["rope_base"] = rescale_theta(
    LLAMA31_CONFIG_8B["rope_base"],
    old_context_length,
    LLAMA31_CONFIG_8B["context_length"]
)

# 打印新的 RoPE theta 值
print("New RoPE theta:", LLAMA31_CONFIG_8B["rope_base"])

# 释放内存
del model  # 删除模型以释放内存

gc.collect()  # 运行 Python 垃圾回收器

# 如果可用，清空 CUDA 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 下载 tokenizer 文件
tokenizer_file_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.1-8B",  # Hugging Face Hub 仓库 ID
    filename="original/tokenizer.model",  # tokenizer 文件名
    local_dir="Llama-3.1-8B"  # 本地保存路径
)

# 创建 Tokenizer 实例
tokenizer = Tokenizer(tokenizer_file_path)

# 初始化 Llama3 模型
model = Llama3Model(LLAMA31_CONFIG_8B)

# 计算模型的总参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")  # 打印总参数数量

combined_weights = {}  # 初始化一个字典用于存储合并后的权重

# 循环下载和合并模型的权重文件
for i in range(1, 5):
    # 下载当前权重文件
    weights_file = hf_hub_download(
        repo_id="meta-llama/Llama-3.1-8B",
        filename=f"model-0000{i}-of-00004.safetensors",  # 权重文件名
        local_dir="Llama-3.1-8B"  # 本地保存路径
    )
    # 加载当前权重文件
    current_weights = load_file(weights_file)
    # 合并当前权重到 combined_weights 字典中
    combined_weights.update(current_weights)

# 将合并的权重加载到模型中
load_weights_into_llama(model, LLAMA31_CONFIG_8B, combined_weights)
# 将模型移动到指定的设备（如 GPU）
model.to(device)
# 删除 combined_weights 字典以释放内存
del combined_weights

# 设置随机种子以确保生成结果可复现
torch.manual_seed(123)

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将输入文本转换为 token IDs
    max_new_tokens=25,  # 生成的最大新 token 数量为 25
    context_size=LLAMA31_CONFIG_8B["context_length"],  # 上下文大小
    top_k=1,  # 采样时选择最高概率的 k 个候选（这里是 1）
    temperature=0.  # 温度设置为 0，表示选择确定性的输出
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # NEW: Larger supported context length
    "emb_dim": 4096,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 32,  # Number of layers
    "hidden_dim": 14_336,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
    "rope_freq": {  # NEW: RoPE frequency scaling
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 2048,  # NEW: Half the embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 16,  # NEW: Half the number of layers
    "hidden_dim": 8192,  # NEW: Almost half the size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to save memory
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,  # NEW: Adjustment of the rescaling factor
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}
# 保存旧的上下文长度
old_context_length = LLAMA32_CONFIG_1B["context_length"]
# 更新新的上下文长度
LLAMA32_CONFIG_1B["context_length"] = 8192

# 使用 rescale_theta 函数更新 rope_base 参数
LLAMA32_CONFIG_1B["rope_base"] = rescale_theta(
    LLAMA32_CONFIG_1B["rope_base"],
    old_context_length,
    LLAMA32_CONFIG_1B["context_length"]
)

# 打印新的 RoPE theta 值
print("New RoPE theta:", LLAMA32_CONFIG_1B["rope_base"])

# 释放内存
del model  # 删除模型以释放内存

gc.collect()  # 运行 Python 垃圾回收器

# 如果可用，清空 CUDA 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 下载 tokenizer 文件
tokenizer_file_path = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",  # Hugging Face Hub 仓库 ID
    filename="original/tokenizer.model",  # tokenizer 文件名
    local_dir="Llama-3.2-1B"  # 本地保存路径
)

# 创建 Tokenizer 实例
tokenizer = Tokenizer(tokenizer_file_path)

# 初始化 Llama3 模型
model = Llama3Model(LLAMA32_CONFIG_1B)

# 计算模型的总参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")  # 打印总参数数量

# 考虑权重共享，计算唯一参数的总数
total_params_normalized = total_params - model.tok_emb.weight.numel()
print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

# 下载权重文件
weights_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B",  # Hugging Face Hub 仓库 ID
    filename=f"model.safetensors",  # 权重文件名
    local_dir="Llama-3.2-1B"  # 本地保存路径
)

# 加载当前权重文件
current_weights = load_file(weights_file)

# 将当前权重加载到模型中
load_weights_into_llama(model, LLAMA32_CONFIG_1B, current_weights)
# 将模型移动到指定的设备（如 GPU）
model.to(device)
# 删除当前权重以释放内存
del current_weights

# 检查权重共享是否正确
print("Weight tying:", torch.equal(model.tok_emb.weight, model.out_head.weight))

# 设置随机种子以确保生成结果可复现
torch.manual_seed(123)

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort", tokenizer).to(device),  # 将输入文本转换为 token IDs
    max_new_tokens=25,  # 生成的最大新 token 数量为 25
    context_size=LLAMA32_CONFIG_1B["context_length"],  # 上下文大小
    top_k=1,  # 采样时选择最高概率的 k 个候选（这里是 1）
    temperature=0.  # 温度设置为 0，表示选择确定性的输出
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))