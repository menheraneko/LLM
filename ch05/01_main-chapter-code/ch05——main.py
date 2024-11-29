import torch
from previous_chapters import GPTModel


# 5.1部分代码
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
} # 参数设置

torch.manual_seed(123) # 种子
model = GPTModel(GPT_CONFIG_124M) # 创建模型
model.eval();  # 切换为评估模式

import tiktoken
from previous_chapters import generate_text_simple

def text_to_token_ids(text, tokenizer): # token到id
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'}) # 获得token id
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 张量化，加入batch维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer): # 解码
    flat = token_ids.squeeze(0) # 移除batch
    return tokenizer.decode(flat.tolist()) # 解码

start_context = "Every effort moves you" # 测试
tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)  # 获取token id

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))  #输出解码结果


inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"],作为输入

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]，作为标准token id 的答案


with torch.no_grad():
    logits = model(inputs) # 获取分布

probas = torch.softmax(logits, dim=-1) # 获取分布概率
print(probas.shape) # 形状

token_ids = torch.argmax(probas, dim=-1, keepdim=True) # 贪心选择
print("Token IDs:\n", token_ids) # 输出结果

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}") # 目标序列解码
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}") # 预测结果解码

text_idx = 0 # 测试token id
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]] # 取第一batch的前三项概率分别属于target 1 的概率
print("Text 1:", target_probas_1) # 输出

text_idx = 1 # 此处为batch 2
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# 计算所有token的log
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas) # 输出

# 取均值
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas) # 输出

neg_avg_log_probas = avg_log_probas * -1 # 负平均对数
print(neg_avg_log_probas) # 输出

# (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape)

# (batch_size, num_tokens)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)  # 扁平化前两个维度，进行展开
targets_flat = targets.flatten()  # 此处同理，得到二维向量

print("Flattened logits:", logits_flat.shape)  # 形状
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat) # 交叉熵loss
print(loss) # 输出

perplexity = torch.exp(loss) # 困惑
print(perplexity) # 输出








import os
import urllib.request

file_path = "the-verdict.txt" # 路径
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt" # 链接

if not os.path.exists(file_path): # 导入文件
    with urllib.request.urlopen(url) as response: # 发送请求
        text_data = response.read().decode('utf-8') # 从url读取
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data) # 本地写入
else: # 存在，直接打开
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read() # 读取


# First 100 characters
print(text_data[:99])
# Last 100 characters
print(text_data[-99:])

total_characters = len(text_data) # 总长度
total_tokens = len(tokenizer.encode(text_data)) # 总token数，此处encoder已经分词，去空过了

print("Characters:", total_characters) # 输出信息
print("Tokens:", total_tokens)

from previous_chapters import create_dataloader_v1

# Train/validation ratio
train_ratio = 0.90 # 训练部分占比
split_idx = int(train_ratio * len(text_data)) # 获取分割位置
train_data = text_data[:split_idx] # 前一部分作为训练集
val_data = text_data[split_idx:] # 后面部分作为验证


torch.manual_seed(123) # 种子

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
) # 创建dataloader，用于加载训练数据

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
) # 同理，验证数据

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]: # 若token长度不够
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]: # 验证token长度不够
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")


print("Train loader:") # 输出训练信息
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:") # 验证信息
for x, y in val_loader:
    print(x.shape, y.shape)


train_tokens = 0
for input_batch, target_batch in train_loader: # 遍历训练集
    train_tokens += input_batch.numel() # 统计token数

val_tokens = 0
for input_batch, target_batch in val_loader: # 同理
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens) # 输出相关信息
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

def calc_loss_batch(input_batch, target_batch, model, device): # 计算batch loss
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) # 转移
    logits = model(input_batch) # 预测分布
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) # 计算交叉熵loss
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None): # 计算loader loss
    total_loss = 0. # loss
    if len(data_loader) == 0:
        return float("nan") # 若数据为0
    elif num_batches is None:
        num_batches = len(data_loader) # 获取batch数
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader)) # 降低batch数以符合loader实际情况
    for i, (input_batch, target_batch) in enumerate(data_loader): # 遍历批次
        if i < num_batches: # 获取batch
            loss = calc_loss_batch(input_batch, target_batch, model, device) # 计算该batch loss
            total_loss += loss.item() # 增加
        else:
            break
    return total_loss / num_batches # 返回平均loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")


model.to(device) # 转移


torch.manual_seed(123) # 种子

with torch.no_grad(): #禁用梯度
    train_loss = calc_loss_loader(train_loader, model, device) # 计算训练集loss
    val_loss = calc_loss_loader(val_loader, model, device) # 验证集

print("Training loss:", train_loss) # 输出
print("Validation loss:", val_loss)












# 5.2部分代码
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer): # 训练
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], [] # 初始化loss和token列表
    tokens_seen, global_step = 0, -1 # 设置初始数据

    # Main training loop
    for epoch in range(num_epochs): # 训练轮数
        model.train()  # 切换训练模式

        for input_batch, target_batch in train_loader: # 获取每batch数据
            optimizer.zero_grad()  # 梯度清0
            loss = calc_loss_batch(input_batch, target_batch, model, device) # 交叉熵
            loss.backward()  # 反向传播
            optimizer.step()  # 迭代器更新
            tokens_seen += input_batch.numel() # 更新当前token
            global_step += 1 # 步数+1

            # Optional evaluation step
            if global_step % eval_freq == 0: # 按batch评估
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter) # 获取评估loss
                train_losses.append(train_loss) # 加入列表
                val_losses.append(val_loss) # 加入列表
                track_tokens_seen.append(tokens_seen) # 加入列表
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}") # 输出信息

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        ) # 生成

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter): # 评估
    model.eval() # 切换评估模式
    with torch.no_grad(): # 禁用梯度
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter) # 训练loss
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter) # 验证loss
    model.train() # 切回
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval() # 切换评估模式
    context_size = model.pos_emb.weight.shape[0] # 上下文大小，通过位置嵌入权重来获得，即总位置数
    encoded = text_to_token_ids(start_context, tokenizer).to(device) # 编码
    with torch.no_grad(): # 禁用梯度
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        ) # 获取预测id
    decoded_text = token_ids_to_text(token_ids, tokenizer) # 解码预测
    print(decoded_text.replace("\n", " "))  # 整理输出格式
    model.train() # 切回


# Note:
# Uncomment the following code to calculate the execution time
# import time
# start_time = time.time()

torch.manual_seed(123) # 种子
model = GPTModel(GPT_CONFIG_124M) # 模型
model.to(device) # 迁移
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) # 优化器信息，使用adam，学习率0.0004

num_epochs = 10 # 10轮训练
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
) # 获取训练信息

# Note:
# Uncomment the following code to show the execution time
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):  # 定义一个函数用于绘制损失图
    fig, ax1 = plt.subplots(figsize=(5, 3))  # 创建一个图形和一个坐标轴，设置图形大小为5x3英寸

    # 绘制训练损失和验证损失与训练轮数的关系
    ax1.plot(epochs_seen, train_losses, label="Training loss")  # 绘制训练损失曲线
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")  # 绘制验证损失曲线，使用虚线样式
    ax1.set_xlabel("Epochs")  # 设置x轴标签为"Epochs"
    ax1.set_ylabel("Loss")  # 设置y轴标签为"Loss"
    ax1.legend(loc="upper right")  # 在右上角显示图例
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 仅在x轴上显示整数刻度

    # 创建第二个x轴用于显示已处理的tokens数量
    ax2 = ax1.twiny()  # 创建一个共享y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 进行一条不可见的绘图，以对齐刻度
    ax2.set_xlabel("Tokens seen")  # 设置第二个x轴的标签为"Tokens seen"

    fig.tight_layout()  # 调整布局以确保图形的美观
    plt.savefig("loss-plot.pdf")  # 将图形保存为PDF文件
    plt.show()  # 显示图形

# 生成一个包含从0到num_epochs的线性间隔的张量，长度为train_losses的长度
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# 调用绘图函数
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)












# 5.3部分代码
model.to("cpu") # 迁移
model.eval() # 评估

tokenizer = tiktoken.get_encoding("gpt2") # tokenizer

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
) # 预测token id

print("Output text:\n", token_ids_to_text(token_ids, tokenizer)) # 解码

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
} # 词表信息

inverse_vocab = {v: k for k, v in vocab.items()} # 逆转词表

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
) # 下一位置的分布

probas = torch.softmax(next_token_logits, dim=0) # 归一化
next_token_id = torch.argmax(probas).item() # 贪心策略

# The next generated token is then as follows:
print(inverse_vocab[next_token_id]) # 输出

torch.manual_seed(123) # 种子
next_token_id = torch.multinomial(probas, num_samples=1).item() # 温度标度法，采样数1
print(inverse_vocab[next_token_id]) # 输出

def print_sampled_tokens(probas): # 改进的函数
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)] #多次采样
    sampled_ids = torch.bincount(torch.tensor(sample)) # 频率
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}") # 输出键值对，为频率和token

print_sampled_tokens(probas) # 输出

def softmax_with_temperature(logits, temperature): # 温度归一化
    scaled_logits = logits / temperature # 温度处理，温度大于1分布更均匀，小于1更突出
    return torch.softmax(scaled_logits, dim=0) # 归一化

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

# Plotting
x = torch.arange(len(vocab)) # 平均长度
bar_width = 0.15  # 宽度

# 创建一个大小为5x3的图形和坐标轴
fig, ax = plt.subplots(figsize=(5, 3))

# 遍历每个温度值及其对应的索引
for i, T in enumerate(temperatures):
    # 为每个温度绘制条形图，条形图的横坐标偏移 bar_width，以避免重叠
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

# 设置y轴标签
ax.set_ylabel('Probability')

# 设置x轴的刻度，使其对应于唯一的标记索引
ax.set_xticks(x)

# 使用词汇表的键作为x轴的刻度标签，并旋转标签以提高可读性
ax.set_xticklabels(vocab.keys(), rotation=90)

# 为图例添加标签，以区分不同温度的条形
ax.legend()

# 调整布局，防止重叠，确保所有元素都适合
plt.tight_layout()

# 将图形保存为名为 "temperature-plot.pdf" 的PDF文件
plt.savefig("temperature-plot.pdf")

# 显示生成的图形
plt.show()

# 使用提供的函数打印第二个温度（索引为1）对应的采样标记
print_sampled_tokens(scaled_probas[1])

# 使用提供的函数打印第三个温度（索引为2）对应的采样标记
print_sampled_tokens(scaled_probas[2])

top_k = 3 # 设置top3
top_logits, top_pos = torch.topk(next_token_logits, top_k) # 获取top3个分布和pos

print("Top logits:", top_logits) # 输出
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits
)  # 把小于topk的分布，均使用inf掩码

print(new_logits) # 输出

topk_probas = torch.softmax(new_logits, dim=0) # 归一化
print(topk_probas) # 输出

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None): # 改进的生成函数

    # 按上下文大小取batch
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad(): # 禁用梯度
            logits = model(idx_cond) # 分布
        logits = logits[:, -1, :] # 贪心

        # 若使用topk采样
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k) # 获取topk
            min_val = top_logits[:, -1] # 最小分布元
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits) # mask

        # 使用温度标度
        if temperature > 0.0:
            logits = logits / temperature # 处理

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len) # 归一化

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)  # 温度标度采样

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1) # 贪心

        if idx_next == eos_id: # 生成结束
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # 记录

    return idx

torch.manual_seed(123) # 种子

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
) # 获取预测id

print("Output text:\n", token_ids_to_text(token_ids, tokenizer)) # 输出









# 5.4部分代码

torch.save(model.state_dict(), "model.pth") # 保存

model = GPTModel(GPT_CONFIG_124M) # 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))  #从路径加载权重
model.eval(); # 切换评估模式

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
) # 保存

checkpoint = torch.load("model_and_optimizer.pth", weights_only=True) # 加载

model = GPTModel(GPT_CONFIG_124M) # 模型
model.load_state_dict(checkpoint["model_state_dict"]) # 加载状态

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1) # 优化器
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # 加载优化器参数
model.train(); # 训练

# 5.5部分代码
# Relative import from the gpt_download.py contained in this folder
# 从 gpt_download 模块中导入下载和加载 GPT-2 的函数
from gpt_download import download_and_load_gpt2

# 下载并加载指定大小的GPT-2模型，并返回设置与参数
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# 打印加载的设置和参数字典的键
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

# 打印 token 嵌入权重张量
print(params["wte"])

# 打印 token 嵌入权重张量的维度
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# 定义不同 GPT-2 模型的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 复制基本配置并更新特定模型设置
model_name = "gpt2-small (124M)"  # 选择的模型名称
NEW_CONFIG = GPT_CONFIG_124M.copy()  # 假设 GPT_CONFIG_124M 是基础配置
NEW_CONFIG.update(model_configs[model_name])  # 更新为特定模型的配置
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})  # 添加上下文长度和偏置

# 实例化 GPTModel，使用更新后的配置
gpt = GPTModel(NEW_CONFIG)
gpt.eval();  # 将模型设置为评估模式


# 定义两个张量赋值的函数
def assign(left, right):
    # 如果左右形状不匹配，则抛出错误
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))  # 返回可训练的参数


import numpy as np  # 导入NumPy库


# 定义加载权重到GPT模型的函数
def load_weights_into_gpt(gpt, params):
    # 赋值位置嵌入权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 赋值token嵌入权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历每个 Transformer 块
    for b in range(len(params["blocks"])):
        # 分割注意力权重
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)

        # 赋值查询、键、值权重
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 分割注意力偏置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 赋值输出投影权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 赋值前馈网络层的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 赋值层归一化的缩放和偏置
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

        # 赋值最终层的规范化和输出头权重
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# 将参数加载到模型中
load_weights_into_gpt(gpt, params)

# 将模型转移到指定设备（如 GPU）
gpt.to(device);

#种子
torch.manual_seed(123)

# 生成文本 token IDs
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),  # 将文本转换为 token IDs
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))