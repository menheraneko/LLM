import urllib.request
import zipfile
import os
from pathlib import Path



# 6.1无





# 6.2部分代码
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path): # 下载解压数据集
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.") # 存在，退出
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response: # 下载
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read()) # 本地保存

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path) # 解压缩

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection" # 设置文件路径
    os.rename(original_file_path, data_file_path) # 重命名
    print(f"File downloaded and saved as {data_file_path}") # 输出信息

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) # 下载解压
import pandas as pd

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"]) # 读取文件


print(df["Label"].value_counts()) # 输出


def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0] # 计算spam的数量

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) # 欠采样，保持分类平衡

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]]) # 重新组合

    return balanced_df


balanced_df = create_balanced_dataset(df) # 欠采样
print(balanced_df["Label"].value_counts()) # 输出结果

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1}) # 映射到0和1


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) # 打乱顺序，随机采样

    # Calculate split indices
    train_end = int(len(df) * train_frac) # 训练集分位点
    validation_end = train_end + int(len(df) * validation_frac) # 验证集分位点

    # Split the DataFrame
    train_df = df[:train_end] # 取训练集
    validation_df = df[train_end:validation_end] # 验证集
    test_df = df[validation_end:] # 测试集

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) # 数据集分割
# Test size is implied to be 0.2 as the remainder

train_df.to_csv("train.csv", index=None) # 分别输出结果
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)



# 6.3部分代码

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # 编码

import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256): # 初始化，特殊设置了填充标签的token id
        self.data = pd.read_csv(csv_file) # 读取数据

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]  # 编码

        if max_length is None: # 若不存在最大长度限制
            self.max_length = self._longest_encoded_length() # 取最序列，作为填充长度
        else:
            self.max_length = max_length # 取最大长度
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] # 取到最大长度为止
                for encoded_text in self.encoded_texts # 逐项处理
            ]


        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ] # 填充剩余长度

    def __getitem__(self, index):
        encoded = self.encoded_texts[index] # 获取索引下的id
        label = self.data.iloc[index]["Label"] # 读取该位置标签
        return (
            torch.tensor(encoded, dtype=torch.long), # id
            torch.tensor(label, dtype=torch.long) # 标签
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self): # 取最长序列
        max_length = 0
        for encoded_text in self.encoded_texts: # 遍历
            encoded_length = len(encoded_text) # 获取当前长度
            if encoded_length > max_length: # 如更长
                max_length = encoded_length # 更新最长长度
        return max_length
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
) # 创建训练数据集

print(train_dataset.max_length) # 输出最长

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
) # 验证集
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
) # 测试集

from torch.utils.data import DataLoader

num_workers = 0 # 工作数
batch_size = 8 # 批次大小

torch.manual_seed(123) # 种子

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
) # loader

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
) # loader

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
) # loader

print("Train loader:") # 输出信息
for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape) # 形状
print("Label batch dimensions", target_batch.shape) # 形状

print(f"{len(train_loader)} training batches") # 输出batch数目
print(f"{len(val_loader)} validation batches") # 验证集
print(f"{len(test_loader)} test batches") # 测试集












# 6.4部分代码

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

from previous_chapters import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)


text_1 = "Every effort moves you"

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))




















# 6.5部分代码
for param in model.parameters():
    param.requires_grad = False # 冻结参数

torch.manual_seed(123) # 种子

num_classes = 2 # 分类数
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes) # 改造输出头，使得输出维度为2，建立二分类任务

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True # 解冻transformer block参数

for param in model.final_norm.parameters():
    param.requires_grad = True # 解冻最后一层归一化参数


inputs = tokenizer.encode("Do you have time") # 编码
inputs = torch.tensor(inputs).unsqueeze(0) # 张量化
print("Inputs:", inputs) # 输出
print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

with torch.no_grad():
    outputs = model(inputs) # 禁用dropout，获得输出

print("Outputs:\n", outputs) # 输出
print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)

print("Last output token:", outputs[:, -1, :]) # 取最后一项的分类情况，因为因果注意力下，该部分包含了所有token信息











# 6.6部分代码

probas = torch.softmax(outputs[:, -1, :], dim=-1) # 归一化获取概率
label = torch.argmax(probas) # 贪心找到分类
print("Class label:", label.item()) # 输出

logits = outputs[:, -1, :] # 取最后一项，因果注意力影响
label = torch.argmax(logits) # 获取分类
print("Class label:", label.item()) # 输出

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval() # 切换评估模型
    correct_predictions, num_examples = 0, 0 # 初始化正确数和总样本数

    if num_batches is None:
        num_batches = len(data_loader) # 获取batch数
    else:
        num_batches = min(num_batches, len(data_loader)) # 取较小者

    for i, (input_batch, target_batch) in enumerate(data_loader): # 遍历data
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device) # 转移

            with torch.no_grad(): # 禁用梯度
                logits = model(input_batch)[:, -1, :]  # 获取最后一项的分布
            predicted_labels = torch.argmax(logits, dim=-1) # 贪心取最大值

            num_examples += predicted_labels.shape[0] # 总处理数+1
            correct_predictions += (predicted_labels == target_batch).sum().item() # 若和target相同，则预测正确
        else:
            break
    return correct_predictions / num_examples # 准确率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# As of this writing, in PyTorch 2.4, the results obtained via CPU and MPS were identical.
# However, in earlier versions of PyTorch, you may observe different results when using MPS.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#print(f"Running on {device} device.")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # 种子

train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10) # 训练集上的准确性
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10) # 验证集
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10) # 测试集

print(f"Training accuracy: {train_accuracy*100:.2f}%") # 输出准确性
print(f"Validation accuracy: {val_accuracy*100:.2f}%") # 验证集
print(f"Test accuracy: {test_accuracy*100:.2f}%") # 测试集

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) # 转移
    logits = model(input_batch)[:, -1, :]  # 获取最后一项分布
    loss = torch.nn.functional.cross_entropy(logits, target_batch) # 交叉熵
    return loss

# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0. # 总loss
    if len(data_loader) == 0:
        return float("nan") # 空数据集，退出
    elif num_batches is None:
        num_batches = len(data_loader) # 默认取loader中的batch
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader)) # 保证batch满足数据集
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device) # 交叉熵
            total_loss += loss.item() # loss增强
        else:
            break
    return total_loss / num_batches # 均值loss


with torch.no_grad(): # 禁用梯度
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5) # 训练loss
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5) # 验证loss
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5) # 测试loss

print(f"Training loss: {train_loss:.3f}") # 输出信息
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")









# 6.7部分代码
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] # 初始化loss列表
    examples_seen, global_step = 0, -1 # 跟踪指标

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # 训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 梯度清0
            loss = calc_loss_batch(input_batch, target_batch, model, device) # 交叉熵
            loss.backward() # 反向传播
            optimizer.step() # 优化器迭代
            examples_seen += input_batch.shape[0] # 当前可视token
            global_step += 1 # 步数+1


            if global_step % eval_freq == 0: # 当达到指定频率，开始评估
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter) # 获取loss
                train_losses.append(train_loss) # 加入列表
                val_losses.append(val_loss) # 加入列表
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}") # 输出loss信息


        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter) # 训练准确率
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter) # 验证准确率
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="") # 输出
        print(f"Validation accuracy: {val_accuracy*100:.2f}%") # 输出
        train_accs.append(train_accuracy) # 加入列表
        val_accs.append(val_accuracy) # 加入列表

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# Same as chapter 5
# 代码无改动
# 代码无改动
# 代码无改动
# 代码无改动
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


import time

# 记录训练开始时间
start_time = time.time()

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 初始化AdamW优化器，设置学习率和权重衰减
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# 定义训练的轮数
num_epochs = 5

# 训练分类器，返回训练损失、验证损失、训练准确率、验证准确率和已处理样本数量
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

# 记录训练结束时间
end_time = time.time()
# 计算并打印训练所用的时间（以分钟为单位）
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入绘图库
import matplotlib.pyplot as plt

# 定义绘图函数，绘制训练和验证损失或准确率
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    # 创建绘图窗口
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失（或准确率）与轮数的关系
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")  # x轴标签
    ax1.set_ylabel(label.capitalize())  # y轴标签
    ax1.legend()  # 显示图例

    # 创建第二个x轴，用于显示已处理样本数量
    ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
    ax2.plot(examples_seen, train_values, alpha=0)  # 绘制不可见的图以对齐刻度
    ax2.set_xlabel("Examples seen")  # 设置第二个x轴标签

    fig.tight_layout()  # 调整布局以留出空间
    plt.savefig(f"{label}-plot.pdf")  # 保存图像为PDF文件
    plt.show()  # 显示图像

# 生成轮数和已处理样本数量的张量
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

# 绘制训练和验证损失
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# 生成轮数和已处理样本数量的张量
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

# 绘制训练和验证准确率
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

# 计算训练、验证和测试集的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

# 打印训练、验证和测试集的准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")




# 6.8部分代码
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256): # 分类任务
    model.eval() # 评估模式


    input_ids = tokenizer.encode(text) # 编码
    supported_context_length = model.pos_emb.weight.shape[0] # 支持的最大上下文长度，即pos总数


    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)] # 避免长度过长，取最大长度作为比较

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids)) # 填充
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # 增加batch维度

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # 获取分布
    predicted_label = torch.argmax(logits, dim=-1).item() # 获取最大概率项

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam" # 返回分类结果


text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
) # 测试文本

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
)) # 输出结果

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
) # 测试文本

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
)) #输出结果

torch.save(model.state_dict(), "review_classifier.pth") # 保存

model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True) # 加载参数
model.load_state_dict(model_state_dict) # 加载到模型