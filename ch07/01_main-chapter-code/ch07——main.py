import json
import os
import urllib


# 7.1部分无代码








# 7.2部分代码
def download_and_load_file(file_path, url): # 加载文件

    if not os.path.exists(file_path): # 文件路径不存在
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8") # 通过url获取
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data) # 本地写入
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read() # 存在，则读取

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file) # 加载json

    return data


file_path = "instruction-data.json" # 文件路径
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
) # url

data = download_and_load_file(file_path, url) # 获取训练数据
print("Number of entries:", len(data)) # 数据长度

print("Example entry:\n", data[50]) # 样例数

print("Another example entry:\n", data[999]) # 其他样例

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    ) # 规范化输出，形成prompt

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" # 加入输入

    return instruction_text + input_text # 组合指令和输入


model_input = format_input(data[50]) # 获取模型输入
desired_response = f"\n\n### Response:\n{data[50]['output']}" # 获取预期输出

print(model_input + desired_response) # 输出

model_input = format_input(data[999]) # 获取模型输入
desired_response = f"\n\n### Response:\n{data[999]['output']}" # 获取预期输出

print(model_input + desired_response) # 输出

train_portion = int(len(data) * 0.85)  # 分割数据集85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # 剩下为验证

train_data = data[:train_portion] # 获取训练数据
test_data = data[train_portion:train_portion + test_portion] # 测试
val_data = data[train_portion + test_portion:] # 验证
print("Training set length:", len(train_data)) # 输出信息
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))












# 7.3部分代码
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer): # 初始化
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = [] # tokenize列表
        for entry in data:
            instruction_plus_input = format_input(entry) # 获取prompt
            response_text = f"\n\n### Response:\n{entry['output']}" # 预期输出
            full_text = instruction_plus_input + response_text # 整体输入
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            ) # 编码

    def __getitem__(self, index):
        return self.encoded_texts[index] # 返回id

    def __len__(self):
        return len(self.data)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # 输出编码序列


def custom_collate_draft_1(
    batch, # 批次
    pad_token_id=50256, # 填充标签id
    device="cpu" # 设备
):
    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item)+1 for item in batch) # 取最大长度


    inputs_lst = [] # 初始化输入队列

    for item in batch:
        new_item = item.copy() # 复制

        new_item += [pad_token_id] # 加入填充

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        ) # 填充到max

        inputs = torch.tensor(padded[:-1]) # 去除最后多余的填充
        inputs_lst.append(inputs) # 添加列表


    inputs_tensor = torch.stack(inputs_lst).to(device) # 折叠，创建batch维度
    return inputs_tensor

# 测试示例
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
) # 批次数据

print(custom_collate_draft_1(batch)) # batch处理

def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch) # 获取最大长度

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], [] # 初始化输入和目标队列

    for item in batch:
        new_item = item.copy() # 复制

        new_item += [pad_token_id] # 填充

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        ) # 填充到最大长度
        inputs = torch.tensor(padded[:-1])  # 去除多余填充
        targets = torch.tensor(padded[1:])  # 目标序列多移位1
        inputs_lst.append(inputs) # 添加结果
        targets_lst.append(targets) # 添加结果

    # 折叠，创建batch维度
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch) # 获取输入和目标数据

# 输出
print(inputs)
print(targets)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,       #
    allowed_max_length=None, # 长度限制
    device="cpu"
):

    batch_max_length = max(len(item)+1 for item in batch) # 获取最大长度


    inputs_lst, targets_lst = [], [] # 初始化列表

    for item in batch:
        new_item = item.copy() # 复制

        new_item += [pad_token_id] # 填充

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        ) # 填充到最大长度
        inputs = torch.tensor(padded[:-1])  # 去除多余填充
        targets = torch.tensor(padded[1:])  # 目标序列移位
        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id # 获取除第一个外的所有填充标签
        indices = torch.nonzero(mask).squeeze() # 提取填充token
        if indices.numel() > 1: # 若存在多个填充
            targets[indices[1:]] = ignore_index # 除第一个外，全部mask


        if allowed_max_length is not None: # 若有长度限制
            inputs = inputs[:allowed_max_length] # 取到最大长度
            targets = targets[:allowed_max_length] # 取到最大长度

        inputs_lst.append(inputs) # 加入列表
        targets_lst.append(targets) # 加入列表

    # 折叠创建batch维
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch) # 获取输入和目标数据
print(inputs) # 输出
print(targets)


logits_1 = torch.tensor(
    [[-1.0, 1.0],  # 1st training example
     [-0.5, 1.5]]  # 2nd training example
) # 测试样例
targets_1 = torch.tensor([0, 1]) # 目标


loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1) # 交叉熵计算
print(loss_1) # 输出


logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  # New 3rd training example
) # 测试样例
targets_2 = torch.tensor([0, 1, 1]) # 目标

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2) # 交叉熵
print(loss_2) # 输出


targets_3 = torch.tensor([0, 1, -100]) # 目标3

loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3) # 交叉熵
print(loss_3) # 输出
print("loss_1 == loss_3:", loss_1 == loss_3) # 验证是否相同










# 7.4部分代码

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is much faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")

print("Device:", device)

from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
) # 使用标准库


from torch.utils.data import DataLoader


# 参数设置
num_workers = 0 # 工作数
batch_size = 8 # 批次大小

torch.manual_seed(123) # 种子

train_dataset = InstructionDataset(train_data, tokenizer) # 获取dataset
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
) # 获取loader

val_dataset = InstructionDataset(val_data, tokenizer) # 验证集
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
) # loader

test_dataset = InstructionDataset(test_data, tokenizer) # 测试集
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
) # loader

print("Train loader:") # 输出相关信息
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)


print(inputs[0]) # 输出input
print(targets[0]) # target















# 7.5部分代码
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


# 基本配置字典，定义模型的基本参数
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # dropout 率
    "qkv_bias": True         # 是否使用查询-key-value 的偏置
}

# 不同模型的配置字典，包含每个模型的具体参数
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小模型
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # 中等模型
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 大模型
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},     # 超大模型
}

# 选择要使用的模型
CHOOSE_MODEL = "gpt2-medium (355M)"

# 更新基本配置，将选择的模型配置合并到基本配置中
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 从模型名称中提取出模型大小，去掉括号及其内容
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# 下载并加载指定模型的配置和参数
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"  # 指定模型存储目录
)

# 创建 GPT 模型实例，使用更新后的基本配置
model = GPTModel(BASE_CONFIG)

# 加载模型权重到模型实例中
load_weights_into_gpt(model, params)

# 将模型设置为评估模式
model.eval()

# 设置随机种子，以确保结果可复现
torch.manual_seed(123)

# 格式化输入文本
input_text = format_input(val_data[0])
print(input_text)  # 打印格式化后的输入文本

# 从之前的章节导入生成、文本到标记 ID 以及标记 ID 到文本的转换函数
from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# 使用模型生成新的文本标记 ID
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),  # 将输入文本转换为标记 ID
    max_new_tokens=35,  # 最大生成的新标记数
    context_size=BASE_CONFIG["context_length"],  # 上下文长度
    eos_id=50256,       # 结束标记 ID
)

# 将生成的标记 ID 转换为文本格式
generated_text = token_ids_to_text(token_ids, tokenizer)

# 处理生成的文本，提取出模型生成的响应部分
response_text = (
    generated_text[len(input_text):]  # 从生成文本中切掉输入文本部分
    .replace("### Response:", "")      # 移除响应标记
    .strip()                           # 去掉前后的空格
)

# 打印最终的响应文本
print(response_text)








# 7.6部分代码
from previous_chapters import (
    calc_loss_loader,
    train_model_simple
)

model.to(device) # 转移

torch.manual_seed(123) # 种子

with torch.no_grad(): # 禁用梯度
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5) # 计算loss，训练
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5) # 验证

print("Training loss:", train_loss) # 输出信息
print("Validation loss:", val_loss)

import time

# 记录开始时间，用于计算训练时间
start_time = time.time()

# 设置随机种子，以确保结果可复现
torch.manual_seed(123)

# 初始化 AdamW 优化器，设置学习率和权重衰减
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

# 定义训练的轮数
num_epochs = 2

# 训练模型，记录训练损失、验证损失和已处理的标记数
train_losses, val_losses, tokens_seen = train_model_simple(
    model,                # 要训练的模型
    train_loader,         # 训练数据加载器
    val_loader,           # 验证数据加载器
    optimizer,            # 优化器
    device,               # 设备（CPU或GPU）
    num_epochs=num_epochs, # 训练轮数
    eval_freq=5,         # 每隔多少轮进行一次评估
    eval_iter=5,         # 每次评估时的迭代次数
    start_context=format_input(val_data[0]),  # 评估时的初始上下文
    tokenizer=tokenizer   # 词汇表分词器
)

# 记录结束时间
end_time = time.time()

# 计算训练所用的时间（以分钟为单位）
execution_time_minutes = (end_time - start_time) / 60
# 打印训练完成的时间
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 从之前的章节导入绘图函数
from previous_chapters import plot_losses

# 创建一个张量，表示训练轮数的范围
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

# 绘制训练损失和验证损失的图形
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)











# 7.7部分代码
torch.manual_seed(123) # 种子


for entry in test_data[:3]: # 遍历到第3列数据

    input_text = format_input(entry) # 规范化输出

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    ) # 通过input的token id 生成文本

    generated_text = token_ids_to_text(token_ids, tokenizer) # 解码
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
) # 把### Response:"进行替换，并从回答中分离

    print(input_text) # 输出
    print(f"\nCorrect response:\n>> {entry['output']}") # 正确回答
    print(f"\nModel response:\n>> {response_text.strip()}") # 预测回答
    print("-------------------------------------")

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry) # 规范化输出

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    ) # 通过input的token id 生成文本

    generated_text = token_ids_to_text(token_ids, tokenizer) # 预测文本解码
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip() # 分离回答

    test_data[i]["model_response"] = response_text # 用作测试集


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

print(test_data[0])


import re


file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth" # 文件名
torch.save(model.state_dict(), file_name) # 保存模型参数到路径
print(f"Model saved as {file_name}") # 输出

# Load model via 加载模型
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))







# 7.8部分代码
import psutil

import psutil  # 导入 psutil 库，用于系统和进程管理

def check_if_running(process_name):  # 定义一个函数，用于检查指定进程是否正在运行
    running = False  # 初始化变量，表示进程是否在运行
    for proc in psutil.process_iter(["name"]):  # 遍历所有正在运行的进程，获取进程名称
        if process_name in proc.info["name"]:  # 检查指定进程名称是否在当前进程列表中
            running = True  # 如果找到了指定的进程，设置 running 为 True
            break  # 找到后退出循环
    return running  # 返回进程是否在运行的状态

# 调用函数检查 "ollama" 进程是否正在运行
ollama_running = check_if_running("ollama")

# 如果 "ollama" 进程没有运行，则引发运行时错误
if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

# 打印 "ollama" 进程是否正在运行的状态
print("Ollama running:", check_if_running("ollama"))


# This cell is optional; it allows you to restart the notebook
# and only run section 7.7 without rerunning any of the previous code
import json
from tqdm import tqdm

file_path = "instruction-data-with-response.json" # 文件路径

with open(file_path, "r") as file:
    test_data = json.load(file) # 加载文件


def format_input(entry): # 规范化输出，写入prompt
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    ) # 指令文本

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" # 输入文本

    return instruction_text + input_text # 模型输入数据


import urllib.request

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
): # 查询模型
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {    # 相关参数
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    } # 以字典方式常创建数据


    # 转json并同时编码
    payload = json.dumps(data).encode("utf-8")


    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    ) # 用于发起post的设置
    request.add_header("Content-Type", "application/json") # 请求头

    # 回应数据
    response_data = ""
    with urllib.request.urlopen(request) as response: # 获取response
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8") # 逐行解码
            if not line:
                break
            response_json = json.loads(line) # 加载到json
            response_data += response_json["message"]["content"] # 加入响应数据

    return response_data


model = "llama3" # 使用llama3
result = query_model("What do Llamas eat?", model) # 查询，得到结果
print(result) # 输出结果

for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    ) # prompt格式，通过lama打分

    # 输出相关信息
    print("\nDataset response:")
    print(">>", entry['output']) # 完整输出
    print("\nModel response:")
    print(">>", entry["model_response"]) # 预测回答
    print("\nScore:")
    print(">>", query_model(prompt)) # lama返回的分数
    print("\n-------------------------")


def generate_model_scores(json_data, json_key, model="llama3"): # 生成模型得分
    scores = [] # 初始化分数列表
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        ) # prompt设置
        score = query_model(prompt, model) # 获得一系列分数
        try:
            scores.append(int(score)) # 加入列表
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


scores = generate_model_scores(test_data, "model_response") # 评估模型
print(f"Number of scores: {len(scores)} of {len(test_data)}") # 分数总数
print(f"Average score: {sum(scores)/len(scores):.2f}\n") # 平均分

