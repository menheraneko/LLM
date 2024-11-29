from pathlib import Path
import pandas as pd
from previous_chapters import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split
)
import torch
import tiktoken
from previous_chapters import SpamDataset




# e1无代码




# e2
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" # url
zip_path = "sms_spam_collection.zip" # zip文件路径
extracted_path = "sms_spam_collection" # 精确路径
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv" # 合成出完整路径

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) # 下载模型

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"]) # 读取文件，同时分词
balanced_df = create_balanced_dataset(df) # 平衡数据集
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1}) # 标签映射到0，1

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) # 分离训练集，验证集，测试集。比例在0.7，0.2，0.1
train_df.to_csv("train.csv", index=None) # 转换文件类型，存为csv
validation_df.to_csv("validation.csv", index=None) # 验证集
test_df.to_csv("test.csv", index=None) # 测试集



tokenizer = tiktoken.get_encoding("gpt2") # tokenizer
train_dataset = SpamDataset("train.csv", max_length=None, tokenizer=tokenizer) # 创建dataset
val_dataset = SpamDataset("validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer) # 验证集
test_dataset = SpamDataset("test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer) # 测试集

from torch.utils.data import DataLoader

num_workers = 0 # 子线程数
batch_size = 8 # 批次大小

torch.manual_seed(123) # 种子

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
) # 训练集loader

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
) # 验证集loader

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
) # 测试集loader

print("Train loader:") # 输出
for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape) # input形状
print("Label batch dimensions", target_batch.shape) # target形状

# 输出数据集batch数
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")



# e3
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


CHOOSE_MODEL = "gpt2-small (124M)" # 154M模型
INPUT_PROMPT = "Every effort moves" # prompt

BASE_CONFIG = {
    "vocab_size": 50257,     # 词表长度
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # attention是否加入bias
} # 基础参数设置

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
} # 模型和各类参数设置

BASE_CONFIG.update(model_configs[CHOOSE_MODEL]) # 更新参数设置

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")") # 模型大小
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2") # 下载并载入模型

model = GPTModel(BASE_CONFIG) # 创建模型
load_weights_into_gpt(model, params) # 加载参数
model.eval() # 评估模式

from previous_chapters import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)


text_1 = "Every effort moves you" # 测试示例1

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
) # 生成预测token id

print(token_ids_to_text(token_ids, tokenizer)) # 解码输出

torch.manual_seed(123) # 种子

num_classes = 2 # 二分类任务
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes) # 模型输出层设置，分类头

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 1.2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")

model.to(device);  # 转移
from previous_chapters import calc_accuracy_loader


torch.manual_seed(123) # 种子

# 计算准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10) # 训练集
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10) # 验证集
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10) # 测试集

# 输出准确率信息
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")








# e4
import math

class LoRALayer(torch.nn.Module): # LORA层
    def __init__(self, in_dim, out_dim, rank, alpha): # 初始化
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank)) # 初始化A为0
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # 均匀分布初始化
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)) # 初始化B为0
        self.alpha = alpha # 缩放因子

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B) # 计算LoRA
        return x

class LinearWithLoRA(torch.nn.Module):  #应用到线性层
    def __init__(self, linear, rank, alpha): # 初始化
        super().__init__()
        self.linear = linear # 传入线形层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        ) # 处理

    def forward(self, x):
        return self.linear(x) + self.lora(x) # 应用LoRA

def replace_linear_with_lora(model, rank, alpha): # 替换linear
    for name, module in model.named_children(): # 取所有layer
        if isinstance(module, torch.nn.Linear): # 若是线性层

            setattr(model, name, LinearWithLoRA(module, rank, alpha)) # 替换
        else: # 若不是
            replace_linear_with_lora(module, rank, alpha) # 对子模块继续应用


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 计算总参数数量，需要梯度
print(f"Total trainable parameters before: {total_params:,}") # 输出信息

for param in model.parameters():
    param.requires_grad = False # 禁用梯度

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 再次计算
print(f"Total trainable parameters after: {total_params:,}")


replace_linear_with_lora(model, rank=16, alpha=16) # 替换linear

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 参数求和
print(f"Total trainable LoRA parameters: {total_params:,}") # 输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # 转移

print(model) # 输出模型信息

torch.manual_seed(123) # 种子

# 计算准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10) # 训练集
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10) # 验证
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10) # 测试

# 输出准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

from previous_chapters import train_classifier_simple


import time  # 导入时间模块，用于计算训练时间

start_time = time.time()  # 记录训练开始的时间

torch.manual_seed(123)  # 设置随机种子，以确保结果的可重复性

# 创建 AdamW 优化器，指定模型的参数、学习率和权重衰减
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5  # 设置训练的总轮数
# 调用训练函数，传入模型、训练数据加载器、验证数据加载器、优化器、设备等参数
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()  # 记录训练结束的时间
# 计算训练的总时间，并转换为分钟
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")  # 打印训练完成的时间

from previous_chapters import plot_values  # 从之前的章节导入绘图函数

# 创建一个张量，用于表示每个 epoch 的编号
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# 创建一个张量，用于表示训练过程中看到的样本数量
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

# 绘制损失值的变化图，包括训练损失和验证损失
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

# 计算训练集的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device)
# 计算验证集的准确率
val_accuracy = calc_accuracy_loader(val_loader, model, device)
# 计算测试集的准确率
test_accuracy = calc_accuracy_loader(test_loader, model, device)

# 打印训练集的准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")
# 打印验证集的准确率
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# 打印测试集的准确率
print(f"Test accuracy: {test_accuracy*100:.2f}%")