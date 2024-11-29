from importlib.metadata import version

import matplotlib
import tiktoken
import torch
import torch.nn as nn



# 4.1部分代码
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
} # 124m模型相关参数展示

class DummyGPTModel(nn.Module):
    def __init__(self, cfg): # 初始化，传入参数
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # token嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # dropout率

        # sequential获取n个block
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) # 最终归一化
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        ) # 获取词表预测结果


    def forward(self, in_idx): # 前向传播
        batch_size, seq_len = in_idx.shape # 获取参数，批次大小和序列长度
        tok_embeds = self.tok_emb(in_idx) # 词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # 位置嵌入
        x = tok_embeds + pos_embeds # input
        x = self.drop_emb(x) # dropout
        x = self.trf_blocks(x) # block计算
        x = self.final_norm(x) # 归一化
        logits = self.out_head(x) # 获取分布
        return logits


# 简单transformer块
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg): # 默认初始化
        super().__init__()
        # A simple placeholder

    def forward(self, x): # 不进行任何操作
        # This block does nothing and just returns its input.
        return x


# 简单层归一化
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5): # 初始化为最小参数
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x): # 不做操作
        # This layer does nothing and just returns its input.
        return x



tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer

batch = [] # 测试

txt1 = "Every effort moves you" # 文本
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1))) # 加入batch
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0) # 第一批次
print(batch)

torch.manual_seed(123) # 种子
model = DummyGPTModel(GPT_CONFIG_124M) # 获取模型对象

logits = model(batch) # 获取结果
print("Output shape:", logits.shape) # 形状
print(logits) # 输出









# 4.2部分代码
torch.manual_seed(123) # 种子

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5) # 测试样例

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) # 简单层和激活函数
out = layer(batch_example) # 输出
print(out)

mean = out.mean(dim=-1, keepdim=True) # 求均值
var = out.var(dim=-1, keepdim=True) # 求方差

print("Mean:\n", mean) # 输出
print("Variance:\n", var)

out_norm = (out - mean) / torch.sqrt(var) # 正态化
print("Normalized layer outputs:\n", out_norm) # 输出结果

mean = out_norm.mean(dim=-1, keepdim=True) # 取平均
var = out_norm.var(dim=-1, keepdim=True) # 方差
print("Mean:\n", mean) # 输出
print("Variance:\n", var) # 输出

torch.set_printoptions(sci_mode=False) # 禁用科学计数
print("Mean:\n", mean) # 输出
print("Variance:\n", var)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim): # 初始化
        super().__init__()
        self.eps = 1e-5 # 避免0作为除数
        self.scale = nn.Parameter(torch.ones(emb_dim)) # 缩放
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 移位

    def forward(self, x): # 前向传播
        mean = x.mean(dim=-1, keepdim=True) # 均值
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 方差，不使用贝塞尔矫正
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # 层归一化
        return self.scale * norm_x + self.shift

ln = LayerNorm(emb_dim=5) # 创建对象
out_ln = ln(batch_example) # 获取结果

mean = out_ln.mean(dim=-1, keepdim=True) # 均值
var = out_ln.var(dim=-1, unbiased=False, keepdim=True) # 方差

print("Mean:\n", mean) # 输出
print("Variance:\n", var)











# 4.3部分代码
class GELU(nn.Module): # 激活函数实现
    def __init__(self): # 初始化
        super().__init__()

    def forward(self, x): # 前向传播
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        )) # 低精度实现

import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU() # 获取库中现有激活函数

# Some sample data
x = torch.linspace(-3, 3, 100) # 测试示例
y_gelu, y_relu = gelu(x), relu(x) # 获取激活结果

plt.figure(figsize=(8, 3)) # 设置图形窗口
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1): # 取样本
    plt.subplot(1, 2, i) # 布局
    plt.plot(x, y) # 绘制
    plt.title(f"{label} activation function") # 设置标题
    plt.xlabel("x") # x
    plt.ylabel(f"{label}(x)") # y
    plt.grid(True) # 添加网格线

# 显示
plt.tight_layout()
plt.show()



class FeedForward(nn.Module):
    def __init__(self, cfg): # 初始化
        super().__init__()
        self.layers = nn.Sequential(  #简单网络，使用GELU激活，包括两个线性层
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x) # 直接计算

print(GPT_CONFIG_124M["emb_dim"]) # 输出嵌入维度

ffn = FeedForward(GPT_CONFIG_124M) # 创建前馈对象

# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) # 示例
out = ffn(x) # 获取结果
print(out.shape) # 输出









# 4.4部分代码

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut): # 初始化
        super().__init__()
        self.use_shortcut = use_shortcut # 快捷连接
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ]) # 定义一系列网络结构

    def forward(self, x): # 前向传播
        for layer in self.layers:

            layer_output = layer(x) # 获取输出

            if self.use_shortcut and x.shape == layer_output.shape: # 若当前使用shortcut，且形状符合
                x = x + layer_output # 添加输出作为新输入一部分
            else: # 正常传播
                x = layer_output
        return x


def print_gradients(model, x):
    # Forward pass
    output = model(x) # 获取结果
    target = torch.tensor([[0.]]) # 示例

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss() # mse loss
    loss = loss(output, target) # 计算loss

    loss.backward() # 反向传播

    for name, param in model.named_parameters():
        if 'weight' in name: # 找到各层权重

            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}") # 输出


layer_sizes = [3, 3, 3, 3, 3, 1] # 设置维度

sample_input = torch.tensor([[1., 0., -1.]]) # 示例

torch.manual_seed(123) # 种子
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
) # 获取对象，不shortcut
print_gradients(model_without_shortcut, sample_input) # 输出权重

torch.manual_seed(123) # 种子
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
) # shortcut
print_gradients(model_with_shortcut, sample_input) # 输出权重











# 4.5部分代码
from previous_chapters import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg): # 初始化
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]) # 多头注意力模块

        self.ff = FeedForward(cfg) # 前馈网络模块

        self.norm1 = LayerNorm(cfg["emb_dim"]) # 层归一化1
        self.norm2 = LayerNorm(cfg["emb_dim"]) # 层归一化2

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) # dropout

    def forward(self, x): # 前向传播
        # Shortcut connection for attention
        shortcut = x
        x = self.norm1(x) # 先进行层归一化
        x = self.att(x)  # 计算注意力 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x) # dropout
        x = x + shortcut  # 剩余部分作为快捷连接

        # Shortcut connection for feed forward block
        shortcut = x # 记录
        x = self.norm2(x) # 层归一化2
        x = self.ff(x) # 计算前馈网络
        x = self.drop_shortcut(x) # dropout
        x = x + shortcut  # 连接

        return x


torch.manual_seed(123) # 种子

x = torch.rand(2, 4, 768)  # 测试[batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M) # 获取对象
output = block(x)  # 获取输出

print("Input shape:", x.shape) # 形状
print("Output shape:", output.shape)







# 4.6部分代码

class GPTModel(nn.Module):
    def __init__(self, cfg): # 初始化
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # 词嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"]) # 最后归一化
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        ) # 预测分布

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape # 参数

        # 嵌入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]

        x = self.drop_emb(x) # dropout
        x = self.trf_blocks(x) # 计算
        x = self.final_norm(x) # 归一化
        logits = self.out_head(x) # 预测分布
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M) # 创建对象

out = model(batch) #结果
print("Input batch:\n", batch) # 输出信息
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters()) # 总参数
print(f"Total number of parameters: {total_params:,}") # 输出

print("Token embedding layer shape:", model.tok_emb.weight.shape) # 形状
print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())  #总参数
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

total_size_bytes = total_params * 4 # 字节数

# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)  #转换为mb

print(f"Total size of the model: {total_size_mb:.2f} MB")











# 4.7部分代码

# 贪心解码
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:] # 取最后context_size个数据作为输入

        with torch.no_grad():
            logits = model(idx_cond) # 获取分布


        logits = logits[:, -1, :] # token_num取最后一项，作为测试

        probas = torch.softmax(logits, dim=-1)  # 归一化获得分布的概率(batch, vocab_size)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # 取最大的一项(batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # 把当前序列加入input中 (batch, n_tokens+1)

    return idx



start_context = "Hello, I am" # 测试

encoded = tokenizer.encode(start_context) # token id
print("encoded:", encoded) # 输出

encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 张量化
print("encoded_tensor.shape:", encoded_tensor.shape)  #输出


model.eval() # 更改模型为评估，不用dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)  # 获取结果

print("Output:", out) # 输出
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist()) # 获取token，解码
print(decoded_text) # 输出