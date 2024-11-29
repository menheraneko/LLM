import torch


# 5.1
# 定义词汇表（vocab），键是单词，值是对应的token ID
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
# 创建一个逆向词汇表（inverse_vocab），用于从token ID查找单词
inverse_vocab = {v: k for k, v in vocab.items()}

# 定义下一个token的logits（未归一化的概率分布）
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# 定义输出采样token的函数
def print_sampled_tokens(probas):
    torch.manual_seed(123)  # 设置随机种子以确保结果可复现
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]  # 采样1000个token
    sampled_ids = torch.bincount(torch.tensor(sample))  # 统计每个token的出现频率
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")  # 打印每个token的出现频率及其对应的单词

# 定义带温度的softmax函数
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature  # 使用温度参数缩放logits
    return torch.softmax(scaled_logits, dim=0)  # 返回softmax归一化后的概率分布

# 定义不同的温度值，分别为 1（原始），0.1（较低），5（较高）
temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]  # 计算不同温度下的概率分布

# 输出不同温度下的采样结果
for i, probas in enumerate(scaled_probas):
    print("\n\nTemperature:", temperatures[i])
    print_sampled_tokens(probas)  # 打印采样的token及其频率

# 选取温度5时的索引和'pizza'的索引
temp5_idx = 2  # 温度5的索引
pizza_idx = 6  # 'pizza'在vocab中的索引

# 获取在温度5下'pizza'的概率值
scaled_probas[temp5_idx][pizza_idx]









# 5.2无





#5.3
import tiktoken
import torch
from previous_chapters import GPTModel


# 配置GPT模型参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # 短上下文长度 (orig: 1024)
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # Dropout率
    "qkv_bias": False  # 查询-键-值偏置
}

# 设置随机种子以确保生成过程可复现
torch.manual_seed(123)

# 初始化分词器和模型
tokenizer = tiktoken.get_encoding("gpt2")  # tokenizer
model = GPTModel(GPT_CONFIG_124M)  # 创建模型
model.load_state_dict(torch.load("model.pth", weights_only=True))  # 加载
model.eval()  # 评估

# 从gpt_generate模块导入必要的功能
from gpt_generate import generate, text_to_token_ids, token_ids_to_text
from previous_chapters import generate_text_simple

# 定义起始上下文
start_context = "Every effort moves you"

# 使用简单生成文本的函数
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 无top_k，无温度缩放
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=None,
    temperature=0.0
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 无top_k，无温度缩放
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=None,
    temperature=0.0
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer)) # 输出







# 5.4
import tiktoken
import torch
from previous_chapters import GPTModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

# 设置设备：使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 加载模型和优化器的检查点
checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

# 实例化GPT模型
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型的状态字典
model.to(device)  # 将模型转移到指定设备

# 初始化AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器的状态字典

# 设置模型为训练模式
model.train()

import urllib.request
from previous_chapters import create_dataloader_v1


import os
import urllib.request
import torch

file_path = "the-verdict.txt"  # 路径
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"  #url

# 下载文件，如果不存在
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8') # 网络读取
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)  # 本地保存
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()  # 本地读取

# 训练/验证数据集的比例
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 种子
torch.manual_seed(123)

# 创建训练数据加载器
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

# 创建验证数据加载器
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 从gpt_train导入训练模型的函数
from gpt_train import train_model_simple

num_epochs = 1
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
) # 获取训练数据




# 5.5
import tiktoken
from previous_chapters import GPTModel


import os
import urllib.request
import torch
from gpt_download import download_and_load_gpt2
from gpt_generate import load_weights_into_gpt
from previous_chapters import create_dataloader_v1

# 定义模型配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)  # 种子
tokenizer = tiktoken.get_encoding("gpt2")  # tokenizer
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")  # 获取设置和参数

# 定义模型名称和更新配置
model_name = "gpt2-small (124M)"
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 更新模型配置并实例化模型
NEW_CONFIG = GPT_CONFIG_124M.copy()  # 复制
NEW_CONFIG.update(model_configs[model_name])  # 更新
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})   # 更新

gpt = GPTModel(NEW_CONFIG)
gpt.eval();  # 评估

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_weights_into_gpt(gpt, params)
gpt.to(device);

# 下载文本数据
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# 此处同理，不解释
# 此处同理，不解释
# 此处同理，不解释
# 此处同理，不解释
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# 划分训练和验证集
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 创建数据加载器
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
) # 验证

from gpt_train import calc_loss_loader

torch.manual_seed(123) # 种子
train_loss = calc_loss_loader(train_loader, gpt, device) # 训练loss
val_loss = calc_loss_loader(val_loader, gpt, device) # 验证loss

print("Training loss:", train_loss) # 输出
print("Validation loss:", val_loss)

settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2") # 设置和参数

model_name = "gpt2-xl (1558M)" # 名称
NEW_CONFIG = GPT_CONFIG_124M.copy() # 复制
NEW_CONFIG.update(model_configs[model_name])  #更新
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG) # 模型
gpt.eval() # 评估

load_weights_into_gpt(gpt, params)  # 加载权重
gpt.to(device) # 转移

torch.manual_seed(123) # 种族
train_loss = calc_loss_loader(train_loader, gpt, device) # 训练loss
val_loss = calc_loss_loader(val_loader, gpt, device) # 验证loss

print("Training loss:", train_loss)
print("Validation loss:", val_loss)









# 5.6
import tiktoken
import torch
from previous_chapters import GPTModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
} # 参数设置


tokenizer = tiktoken.get_encoding("gpt2") # tokenizer
from gpt_download import download_and_load_gpt2
from gpt_generate import load_weights_into_gpt


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
} # 参数设置

model_name = "gpt2-xl (1558M)" # 名称
NEW_CONFIG = GPT_CONFIG_124M.copy() # 复制
NEW_CONFIG.update(model_configs[model_name]) # 加载
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG) # 模型
gpt.eval() # 评估

settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")  #设置和参数
load_weights_into_gpt(gpt, params) # 加载权重


from gpt_generate import generate, text_to_token_ids, token_ids_to_text
torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
) # 预测token id

print("Output text:\n", token_ids_to_text(token_ids, tokenizer)) # 解码