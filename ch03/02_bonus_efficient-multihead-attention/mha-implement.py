import torch
import torch.nn as nn
from packaging.version import parse as parse_version

import torch



torch.manual_seed(123) # 设置种子

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 使用gpu

batch_size = 8 # 参数设置
context_len = 1024 # 上下文长度
embed_dim = 768 # 嵌入维度
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device) # 建立随机张量

# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class Ch03_MHA_Wrapper(nn.Module): # 此处仅为重定义的模型，具体实现没有改动

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)


mha_ch03_wrapper = Ch03_MHA_Wrapper(
    d_in=embed_dim,
    d_out=embed_dim//12,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 创建对象，设置12个头

out = mha_ch03_wrapper(embeddings) # 获取上下文向量
print(out.shape) # 输出结果




# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class Ch03_MHA(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


mha_ch03 = Ch03_MHA(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 创建多头注意力模块，设置12个头

out = mha_ch03(embeddings) # 获取上下文向量
print(out.shape) # 输出结果













# 此处主体部分未做改动，仅对改动部分进行解释
# 此处主体部分未做改动，仅对改动部分进行解释
# 此处主体部分未做改动，仅对改动部分进行解释
# 此处主体部分未做改动，仅对改动部分进行解释
class MultiHeadAttentionCombinedQKV(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias) # 将QKV三层线性变化层统一到一个线性层，默认不使用bias
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # 计算得到联合的qkv矩阵。相当于在原来基础上，cat了dim=-1   (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # 展开qkv矩阵   (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # 重排列张量维度，便于分离  (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离query，key，value  (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # 后面代码相同，不做解释
        # 后面代码相同，不做解释
        # 后面代码相同，不做解释
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        context_vec = context_vec.transpose(1, 2)

        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec


mha_combined_qkv = MultiHeadAttentionCombinedQKV(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 创建对象，使用12个头

out = mha_combined_qkv(embeddings) # 获取上下文向量
print(out.shape) # 形状




import math


class MHAEinsum(nn.Module):
    # 初始化
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保 d_out 可以被 num_heads 整除

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头的数量
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 初始化qkv权重矩阵，没有改动
        self.W_query = nn.Parameter(torch.randn(d_out, d_in))
        self.W_key = nn.Parameter(torch.randn(d_out, d_in))
        self.W_value = nn.Parameter(torch.randn(d_out, d_in))

        if qkv_bias:  # 若使用bias，则初始化bias为0
            self.bias_q = nn.Parameter(torch.zeros(d_out))  # query
            self.bias_k = nn.Parameter(torch.zeros(d_out))  # key
            self.bias_v = nn.Parameter(torch.zeros(d_out))  # value
        else:
            self.register_parameter("bias_q", None)  # 如果不使用则为 None
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        # 此处没有改动
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):  # 初始化参数
        # 使用 Kaiming 均匀分布初始化qkv的权重
        nn.init.kaiming_uniform_(self.W_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_key, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_value, a=math.sqrt(5))

        if self.bias_q is not None:  # 若存在bias
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)  # 计算 fan-in，即输入大小
            bound = 1 / math.sqrt(fan_in)  # 计算边界，确定初始化范围
            nn.init.uniform_(self.bias_q, -bound, bound)  # 初始化查询偏差
            nn.init.uniform_(self.bias_k, -bound, bound)  # 初始化键偏差
            nn.init.uniform_(self.bias_v, -bound, bound)  # 初始化值偏差

    def forward(self, x):
        b, n, _ = x.shape  # 获取输入的批次大小、序列长度和特征维度

        # 计算查询 Q、键 K、值 V，使用 einsum 进行计算，bnd * di = bni维度
        Q = torch.einsum("bnd,di->bni", x, self.W_query)  # query
        K = torch.einsum("bnd,di->bni", x, self.W_key)  # key
        V = torch.einsum("bnd,di->bni", x, self.W_value)  # value

        # 如果使用偏差，将其加到 Q、K 、 V 上
        if self.bias_q is not None:
            Q += self.bias_q  # query
            K += self.bias_k  # key
            V += self.bias_v  # value

        # 为多头注意力重塑 Q、K、V 的形状
        Q = Q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # 重塑并转置查询
        K = K.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # 重塑并转置键
        V = V.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # 重塑并转置值

        # 计算缩放点积注意力
        scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) / (self.head_dim ** 0.5)  # 计算得分并缩放

        # mask
        mask = self.mask[:n, :n].unsqueeze(0).unsqueeze(1).expand(b, self.num_heads, n, n)  # 扩展，保证形状一致
        scores = scores.masked_fill(mask.bool(), -torch.inf)  # 填充

        attn_weights = torch.softmax(scores, dim=-1) # 归一化
        attn_weights = self.dropout(attn_weights)  # dropout

        context_vec = torch.einsum("bhnm,bhmd->bhnd", attn_weights, V)  # 计算上下文向量

        context_vec = context_vec.transpose(1, 2).reshape(b, n, self.d_out)  # 转置并重塑上下文向量
        context_vec = self.out_proj(context_vec)  # 输出

        return context_vec


mha_einsum = MHAEinsum(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 创建对象，使用12个头

out = mha_einsum(embeddings)  # 获取上下文向量
print(out.shape)  # 形状





# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout # 前向传播保证在推理过程中不使用dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True) # 使用scaled_dot_product_attention，包括了计算分数，掩码，权重，归一化的部分

        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec

mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 创建对象，12个注意力头

out = mha_pytorch_scaled(embeddings) # 获取上下文向量
print(out.shape) # 形状





# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
class MHAPyTorchSDPAWithoutFlash(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        if self.context_length >= num_tokens: # 当形状超出预期
            attn_mask = self.mask[:num_tokens, :num_tokens] # 只计算num_token的部分，保持形状一致
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length] # 正常计算

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False) # 统一计算上下文向量

        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec

mha_pytorch_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 创建对象，使用12个头

out = mha_pytorch_sdpa_no_flash(embeddings) # 获取结果
print(out.shape) # 形状








# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
class MHAPyTorchClass(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, need_weights=True):
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out, # 嵌入维度
            num_heads=num_heads, # 头数
            dropout=dropout, # dropout率
            bias=qkv_bias, # 是否使用bias
            add_bias_kv=qkv_bias, # 是否加入bias
            batch_first=True, # 是否保持batch size为第一dim
        ) # 直接使用nn模块下的多头注意力参与计算

        self.need_weights = need_weights # 当该参数设置false时，会使用上面dot的方式计算
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # 类似地统一形状处理
        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[:self.context_length, :self.context_length]

        # 计算上下文向量
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        output = self.proj(attn_output) # 线性变化，整合上下文向量

        return output


mha_pytorch_class_default = MHAPyTorchClass(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device) # 测试对象

out = mha_pytorch_class_default(embeddings) # 获取结果
print(out.shape) # 形状




mha_pytorch_class_noweights = MHAPyTorchClass(
    d_in=embed_dim,
    d_out=embed_dim,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False,
    need_weights=False # NEW!
).to(device)


def normalize_version(version): # 版本检查
    parsed_version = parse_version(version)
    return parse_version(f"{parsed_version.major}.{parsed_version.minor}.{parsed_version.micro}")

current_version = normalize_version(torch.__version__) # 获取版本
MIN_TORCH_VERSION = "2.5.0" # 最低版本
required_version = parse_version(MIN_TORCH_VERSION) # 需要版本

if current_version >= required_version:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask # 版本满足要求，导入


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
# 此处主体部分未作改动，不做重复解释
class MHAPyTorchFlexAttention(nn.Module):

    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        # self.register_buffer("block_mask", create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length))
        # `create_block_mask` function does not support buffers, yet
        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_length, KV_LEN=context_length) # mask，设置因果，且不传入batch和head


    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        # Ensure attn_mask is compatible with expected shape and `batch_first=True`
        # No need to manually adjust for num_heads; ensure it's right for the sequence
        if self.context_length >= num_tokens:
            attn_mask = self.block_mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.block_mask[:self.context_length, :self.context_length]

        context_vec = flex_attention(queries, keys, values, block_mask=attn_mask)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec

if current_version >= required_version and torch.cuda.is_available():

    mha_pytorch_flex = MHAPyTorchFlexAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device) # 测试对象

    out = mha_pytorch_flex(embeddings) # 获取结果
    print(out.shape) # 形状



torch.manual_seed(123) # 种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Running on {device}")

functions = { # 根据不同字符串，通过键值对使用不同的实例化attention对象
    "1) MHA wrapper class": mha_ch03_wrapper,
    "2) MHA Ch03": mha_ch03,
    "3) MHA with combined QKV weights": mha_combined_qkv,
    "4) MHA with Einsum": mha_einsum,
    "5) MHA with PyTorch scaled_dot_product_attention": mha_pytorch_scaled,
    "6) PyTorch's SDPA, no FlashAttention": mha_pytorch_sdpa_no_flash,
    "7) PyTorch MHA class defaults": mha_pytorch_class_default,
    "8) PyTorch MHA with need_weights=False": mha_pytorch_class_noweights
    }

#if current_version >= required_version: # 若版本支持
# functions["8) PyTorch's FlexAttention"] =  mha_pytorch_flex # 使用flex注意力




import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于数据可视化

# 针对黑暗模式美学进行进一步定制
plt.rcParams["figure.facecolor"] = "#121212"  # 设置图表背景颜色为深色
plt.rcParams["axes.facecolor"] = "#121212"    # 设置坐标轴背景颜色为深色
plt.rcParams["axes.edgecolor"] = "white"      # 设置坐标轴边框颜色为白色
plt.rcParams["axes.labelcolor"] = "white"     # 设置坐标轴标签颜色为白色
plt.rcParams["text.color"] = "white"          # 设置文本颜色为白色
plt.rcParams["xtick.color"] = "white"         # 设置 x 轴刻度颜色为白色
plt.rcParams["ytick.color"] = "white"         # 设置 y 轴刻度颜色为白色
plt.rcParams["grid.color"] = "#444444"        # 设置网格线颜色为深灰色
plt.rcParams["lines.linewidth"] = 2            # 设置线条宽度为 2
plt.rcParams["lines.markersize"] = 8           # 设置标记大小为 8

def plot_execution_times(functions, execution_means, execution_stds, filename):
    # 绘制执行时间的函数

    # 创建图表和坐标轴
    fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
    # 绘制柱状图，包含误差条
    bars = ax.bar(functions.keys(), execution_means, yerr=execution_stds, capsize=5, error_kw={'ecolor': 'grey'})

    plt.ylabel("Execution time (ms)")  # 设置 y 轴标签
    plt.xticks(rotation=45, ha="right")  # 设置 x 轴刻度标签的旋转角度和对齐方式

    # 计算新的 y 轴限制，增加一定的边距
    max_execution_time = max(execution_means)  # 获取执行时间的最大值
    upper_ylim = max_execution_time + 0.4 * max_execution_time  # 增加 40% 的边距
    plt.ylim(0, upper_ylim)  # 设置 y 轴的范围

    # 在柱状图上标注执行时间
    for bar in bars:  # 遍历每个柱
        yval = bar.get_height()  # 获取柱的高度（执行时间）
        # 在柱的顶部添加文本标注，显示执行时间
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * upper_ylim), round(yval, 2), ha="center", va="bottom")

    plt.tight_layout()  # 自动调整图形布局
    plt.savefig(filename)  # 保存图形到指定文件
    plt.show()  # 显示图形






import numpy as np

def time_pytorch_function(func, *input, num_repeats=1_000):
    # 定义一个函数，用于测量 PyTorch 函数的执行时间
    start = torch.cuda.Event(enable_timing=True)  # 创建一个开始计时的 CUDA 事件
    end = torch.cuda.Event(enable_timing=True)    # 创建一个结束计时的 CUDA 事件

    # 热身阶段
    for _ in range(5):  # 运行函数 5 次以热身
        func(*input)  # 调用传入的函数
    torch.cuda.synchronize()  # 确保 CUDA 完成所有操作

    times = []  # 用于存储每次运行的时间
    for _ in range(num_repeats):  # 重复测量执行时间
        start.record()  # 记录开始时间
        func(*input)  # 调用传入的函数
        end.record()  # 记录结束时间
        torch.cuda.synchronize()  # 确保 CUDA 完成所有操作
        times.append(start.elapsed_time(end))  # 计算并存储执行时间（毫秒）

    return np.mean(times), np.std(times)  # 返回执行时间的均值和标准差

# 执行每个函数并收集执行时间统计
execution_stats = [time_pytorch_function(fn, embeddings) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats]  # 提取均值
execution_stds = [stat[1] for stat in execution_stats]    # 提取标准差

# 绘制执行时间的柱状图
plot_execution_times(functions, execution_means, execution_stds, filename="1_forward-only.pdf")




def forward_backward(func, embeddings):
    # 定义一个函数，进行前向传播和反向传播
    if embeddings.grad is not None:  # 检查 embeddings 是否已有梯度
        embeddings.grad.zero_()  # 如果有，则将梯度清零

    output = func(embeddings)  # 进行前向传播，计算输出
    loss = output.sum()  # 计算损失，这里简单地求和
    loss.backward()  # 进行反向传播，计算梯度


def time_pytorch_function_forward_backward(func, *input, num_repeats=1_000):
    # 定义一个函数，用于测量前向和反向传播的执行时间
    start = torch.cuda.Event(enable_timing=True)  # 创建一个开始计时的 CUDA 事件
    end = torch.cuda.Event(enable_timing=True)    # 创建一个结束计时的 CUDA 事件

    # 热身阶段
    for _ in range(5):  # 运行函数 5 次以热身
        forward_backward(func, *input)  # 调用前向和反向传播函数
    torch.cuda.synchronize()  # 确保 CUDA 完成所有操作

    times = []  # 用于存储每次运行的时间
    for _ in range(num_repeats):  # 重复测量执行时间
        start.record()  # 记录开始时间
        forward_backward(func, *input)  # 调用前向和反向传播函数
        end.record()  # 记录结束时间
        torch.cuda.synchronize()  # 确保 CUDA 完成所有操作
        times.append(start.elapsed_time(end))  # 计算并存储执行时间（毫秒）

    return np.mean(times), np.std(times)  # 返回执行时间的均值和标准差


# 执行每个函数并收集执行时间统计
execution_stats = [time_pytorch_function_forward_backward(fn, embeddings) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats]  # 提取均值
execution_stds = [stat[1] for stat in execution_stats]    # 提取标准差

# 绘制执行时间的柱状图
plot_execution_times(functions, execution_means, execution_stds, filename="2_forward-and-backward.pdf")




import torch._dynamo  # 导入 PyTorch 的 Dynamo 模块
torch._dynamo.config.suppress_errors = True  # 配置 Dynamo 以抑制错误

def prepare_function(fn):
    # 定义一个函数，将输入的函数编译为可优化的格式
    fn = torch.compile(fn)  # 使用 torch.compile 编译函数
    return fn  # 返回编译后的函数

# 执行每个函数并收集执行时间统计
execution_stats = [time_pytorch_function_forward_backward(prepare_function(fn), embeddings) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats]  # 提取均值
execution_stds = [stat[1] for stat in execution_stats]    # 提取标准差

# 绘制执行时间的柱状图
plot_execution_times(functions, execution_means, execution_stds, filename="3_forward-and-backward-compiled.pdf")