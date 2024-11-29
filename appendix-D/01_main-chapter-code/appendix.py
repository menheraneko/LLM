from importlib.metadata import version
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
} # 模型参数设置

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

torch.manual_seed(123) # 种子
model = GPTModel(GPT_CONFIG_124M) # 创建124M模型
model.eval();  # 切换评估模式

import os
import urllib.request

file_path = "the-verdict.txt" # 文件路径
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt" # url

if not os.path.exists(file_path): # 若不存在文件
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8') # 请求
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data) # 本地写入
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read() # 读取文件

from previous_chapters import create_dataloader_v1

# Train/validation ratio
train_ratio = 0.90 # 划分比例
split_idx = int(train_ratio * len(text_data)) # 计算划分节点


torch.manual_seed(123) # 种子

train_loader = create_dataloader_v1(
    text_data[:split_idx],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
) # 创建训练loader

val_loader = create_dataloader_v1(
    text_data[split_idx:],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
) # 验证loader







# d1
n_epochs = 15 # 训练轮数
initial_lr = 0.0001 # 初始化学习率
peak_lr = 0.01 # 学习率峰值

total_steps = len(train_loader) * n_epochs  #总计算步数
warmup_steps = int(0.2 * total_steps) # 20% warmup
print(warmup_steps) # 输出warmup数

lr_increment = (peak_lr - initial_lr) / warmup_steps # 每次warmup增量

global_step = -1 # 初始化步数
track_lrs = [] # 学习率跟踪列表

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1) # 优化器，使用默认lr

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader: # 遍历数据集
        optimizer.zero_grad() # 梯度清0
        global_step += 1 # 步数+1

        if global_step < warmup_steps: # 若在warmup阶段
            lr = initial_lr + global_step * lr_increment # 学习率增加
        else:
            lr = peak_lr # 设置为峰值

        # 在优化器应用学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr # 设置参数lr为当前学习率
        track_lrs.append(optimizer.param_groups[0]["lr"]) # 第一组参数中lr加入列表

        # Calculate loss and update weights
        # ...

import matplotlib.pyplot as plt

# 创建一个图形对象，设置图形的大小为 5x3 英寸
plt.figure(figsize=(5, 3))

# 设置 y 轴的标签为 "Learning rate" (学习率)
plt.ylabel("Learning rate")

# 设置 x 轴的标签为 "Step" (步骤)
plt.xlabel("Step")

# 计算总的训练步骤数，总步骤数等于每个 epoch 的训练样本数量乘以 epoch 的数量
total_training_steps = len(train_loader) * n_epochs

# 绘制学习率变化曲线，x 轴为步骤数，y 轴为记录的学习率
plt.plot(range(total_training_steps), track_lrs)

# 调整布局以避免标签重叠
plt.tight_layout()

# 将图形保存为名为 "1.pdf" 的文件
plt.savefig("1.pdf")

# 显示绘制的图形
plt.show()









# d2
import math

min_lr = 0.1 * initial_lr # 学习率谷值
track_lrs = [] # 跟踪学习率列表

lr_increment = (peak_lr - initial_lr) / warmup_steps # 学习率增长幅度
global_step = -1 # 总步数

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader: # 遍历数据集
        optimizer.zero_grad() # 梯度清0
        global_step += 1 # 步数+1

        # Adjust the learning rate based on the current phase (warmup or cosine annealing)
        if global_step < warmup_steps: # 若在warmup阶段

            lr = initial_lr + global_step * lr_increment # 线性warmup
        else:
            progress = ((global_step - warmup_steps) /
                        (total_training_steps - warmup_steps)) # 当前步数/总步数，即t/T
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)) # 余弦衰减计算

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr # 应用学习率参数
        track_lrs.append(optimizer.param_groups[0]["lr"]) # 加入跟踪列表


plt.figure(figsize=(5, 3))
# 创建一个图形对象，设置图形的大小为 5x3 英寸
plt.figure(figsize=(5, 3))

# 设置 y 轴的标签为 "Learning rate" (学习率)
plt.ylabel("Learning rate")

# 设置 x 轴的标签为 "Step" (步骤)
plt.xlabel("Step")

# 绘制学习率变化曲线，x 轴为步骤数，y 轴为记录的学习率
plt.plot(range(total_training_steps), track_lrs)

# 调整布局以避免标签重叠
plt.tight_layout()
# 保存为2.pdf
plt.savefig("2.pdf")
# 展示图片
plt.show()











# d3
from previous_chapters import calc_loss_batch

torch.manual_seed(123) # 种子
model = GPTModel(GPT_CONFIG_124M) # 创建124M模型
model.to(device) # 转移

loss = calc_loss_batch(input_batch, target_batch, model, device) # 计算批次loss
loss.backward() # 反向传播

def find_highest_gradient(model): # 找最高梯度
    max_grad = None # 初始化最高梯度
    for param in model.parameters(): # 遍历所有参数
        if param.grad is not None: # 若存在
            grad_values = param.grad.data.flatten() #
            max_grad_param = grad_values.max() # 找最大值
            if max_grad is None or max_grad_param > max_grad: # 若是当前最大值
                max_grad = max_grad_param # 更新最大值
    return max_grad

print(find_highest_gradient(model)) # 输出最大梯度

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，最大值为1
print(find_highest_gradient(model)) # 再次找最大值













# d4
from previous_chapters import evaluate_model, generate_and_print_sample

BOOK_VERSION = True


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6): # 训练函数
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], [] # 初始化训练loss，验证loss，可见token跟踪，学习率跟踪列表
    tokens_seen, global_step = 0, -1 # 初始化可见token数，当前步数

    peak_lr = optimizer.param_groups[0]["lr"] # 获取优化器lr为当前学习率峰值
    total_training_steps = len(train_loader) * n_epochs # 计算总步数
    lr_increment = (peak_lr - initial_lr) / warmup_steps # 计算学习率增长量

    for epoch in range(n_epochs): # 训练轮次
        model.train() # 训练模式
        for input_batch, target_batch in train_loader: # 遍历数据集
            optimizer.zero_grad() # 梯度清0
            global_step += 1 # 当前步数+1

            if global_step < warmup_steps: # 若为warmup阶段
                lr = initial_lr + global_step * lr_increment # 学习率增长
            else:

                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps)) # t/T
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) # 余弦衰减


            for param_group in optimizer.param_groups:
                param_group["lr"] = lr # 更新lr
            track_lrs.append(lr)  # 保存lr到跟踪列表

            loss = calc_loss_batch(input_batch, target_batch, model, device) # 计算批次loss
            loss.backward() # 反向传播

            if BOOK_VERSION:
                if global_step > warmup_steps: # 不是warmup
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            else:
                if global_step >= warmup_steps: # 保证=时也进行裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪

            optimizer.step() # 优化器迭代
            tokens_seen += input_batch.numel() # 计算可见token数

            # Periodically evaluate the model on the training and validation sets
            if global_step % eval_freq == 0: # 若达到评估频率
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                ) # 模型评估
                train_losses.append(train_loss) # 训练loss记录
                val_losses.append(val_loss) # 验证loss
                track_tokens_seen.append(tokens_seen) # 可见token

                print(f"Ep {epoch + 1} (Iter {global_step:06d}): " # 输出loss信息
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        ) # 生成样本

    return train_losses, val_losses, track_tokens_seen, track_lrs



import tiktoken

# Note:
# Uncomment the following code to calculate the execution time
# import time
# start_time = time.time()

torch.manual_seed(123) # 种子
model = GPTModel(GPT_CONFIG_124M) # 创建124M模型
model.to(device) # 转移

peak_lr = 0.001  # lr峰值
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                              weight_decay=0.1)  # 优化器初始化
tokenizer = tiktoken.get_encoding("gpt2") # tokenizer

n_epochs = 15 # 训练轮数
train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
    eval_freq=5, eval_iter=1, start_context="Every effort moves you",
    tokenizer=tokenizer, warmup_steps=warmup_steps,
    initial_lr=1e-5, min_lr=1e-5
) # 训练模型

# Note:
# Uncomment the following code to show the execution time
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

plt.figure(figsize=(5, 3)) # 创建新图像
plt.plot(range(len(lrs)), lrs) # 绘制学习率随步骤变化的曲线，横轴为步骤数（0 到 lrs 列表长度），纵轴为对应的学习率值
plt.ylabel("Learning rate") # y标签
plt.xlabel("Steps") # x标签
plt.show() # 展示

from previous_chapters import plot_losses

epochs_tensor = torch.linspace(1, n_epochs, len(train_losses)) # 创建epoch对应的loss形成的张量，从1到训练轮数
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses) # 绘制loss
plt.tight_layout() # 调整布局，避免重叠
plt.savefig("3.pdf") # 保存为3.png
plt.show() # 展示
