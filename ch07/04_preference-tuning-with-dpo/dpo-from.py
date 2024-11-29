import json


import json  # 导入JSON库用于处理JSON数据
import pprint  # 导入pprint库用于格式化打印

# 读取包含指令和响应的数据文件
file_path = "instruction-data-with-preference.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)  # 加载JSON数据

# 打印数据条目数量
print("Number of entries:", len(data))

# 打印第51条和第1000条数据的详细内容
pprint.pp(data[50])
pprint.pp(data[999])

# 函数：格式化输入
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "  
        f"Write a response that appropriately completes the request."  
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text  # 返回格式化后的文本

# 格式化第51条数据的输入
model_input = format_input(data[50])
print(model_input)

# 打印期望的响应和可能的响应
desired_response = f"### Response:\n{data[50]['chosen']}"
print(desired_response)

possible_response = f"### Response:\n{data[50]['rejected']}"
print(possible_response)

# 划分数据集比例
train_portion = int(len(data) * 0.85)  # 85%用于训练
test_portion = int(len(data) * 0.1)    # 10%用于测试
val_portion = len(data) - train_portion - test_portion  # 剩余5%用于验证

# 划分数据集
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# 打印各数据集的长度
print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

import torch
from torch.utils.data import Dataset

# 自定义数据集类
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 预先对文本进行编码
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)  # 格式化输入
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            # 编码提示和响应
            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            # 保存编码后的文本
            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]  # 返回指定索引的数据

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

# 自定义数据处理函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # 初始化批处理数据
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []
    }

    # 确定最长序列以设置公共填充长度
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # 处理批次中的每个项目
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # 根据公共最大长度调整填充
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()  # 创建掩码

            # 设置填充标记的掩码为False
            mask[len(sequence):] = False

            # 设置输入标记的掩码
            # +2 设置前面两个换行符（"\n"）的掩码为False
            if mask_prompt_tokens:
                mask[:prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # 最终处理
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # 将所有序列堆叠为张量
        tensor_stack = torch.stack(batch_data[key])

        # 可选地截断到最大序列长度
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # 移动到指定设备
        batch_data[key] = tensor_stack.to(device)

    return batch_data  # 返回处理后的批数据


from functools import partial
import torch
import pprint
import tiktoken
from torch.utils.data import DataLoader

# 设置设备为GPU，如果可用的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 使用 functools.partial 来创建一个定制的 collate 函数，
# 便于后续训练和验证时调用
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,            # 将数据直接放在GPU上（如果可用）
    mask_prompt_tokens=True,  # 可选，为了掩码令牌
    allowed_max_length=1024   # 模型支持的上下文长度
)

# 使用示例数据，提取前两个条目进行测试
example_data = data[:2]  # 假设 data 已经被加载

# 打印示例数据
for i in example_data:
    print()
    pprint.pp(i)

# 初始化 token 化工具
tokenizer = tiktoken.get_encoding("gpt2")

# 创建数据集实例
example_dataset = PreferenceDataset(example_data, tokenizer)

# 数据加载器，指定批大小和 collate 函数
example_dataloader = DataLoader(
    example_dataset,
    batch_size=2,
    collate_fn=customized_collate_fn,
    shuffle=False
)

# 遍历数据加载器的一个批次（只取第一个批次）
for batch in example_dataloader:
    break

# 打印批次的键
print("batch.keys:", batch.keys())

# 解码函数，将 token IDs 解码成文本
def decode_tokens_from_batch(token_ids, tokenizer):
    ids_in_python_list = token_ids.flatten().tolist()
    return tokenizer.decode(ids_in_python_list)

# 解码并打印批次中的 prompt
text = decode_tokens_from_batch(
    token_ids=batch["prompt"][0],  # 获取批次中的第一个条目的 prompt
    tokenizer=tokenizer,
)
print(text)

# 解码并打印第一个条目的 chosen 和 rejected 响应
text = decode_tokens_from_batch(
    token_ids=batch["chosen"][0],
    tokenizer=tokenizer,
)
print(text)

text = decode_tokens_from_batch(
    token_ids=batch["rejected"][0],
    tokenizer=tokenizer,
)
print(text)

# 使用遮罩解码选择和拒绝的响应
text = decode_tokens_from_batch(
    token_ids=batch["chosen"][0][batch["chosen_mask"][0]],
    tokenizer=tokenizer,
)
print(text)

text = decode_tokens_from_batch(
    token_ids=batch["rejected"][0][batch["rejected_mask"][0]],
    tokenizer=tokenizer,
)
print(text)

# 创建用于训练、验证和测试的数据加载器
num_workers = 0  # 工作线程数量
batch_size = 8  # 批大小

torch.manual_seed(123)  # 设置随机种子以确保可重复性

# 创建训练集、验证集和测试集的数据集对象和数据加载器
train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = PreferenceDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = PreferenceDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# 打印训练加载器的批次信息
print("Train loader:")
for batch in train_loader:
    print(
        batch["chosen"].shape,
        batch["rejected"].shape,
    )



















import os
from pathlib import Path
import shutil


# 定义微调模型的路径
finetuned_model_path = Path("gpt2-medium355M-sft.pth")
if not finetuned_model_path.exists():
    # 如果模型路径不存在，则尝试在本地找到模型检查点
    relative_path = Path("..") / "01_main-chapter-code" / finetuned_model_path
    if relative_path.exists():
        shutil.copy(relative_path, ".")
    # 如果在Google Colab上运行，从Google Drive文件夹中获取模型
    elif "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        from google.colab import drive
        drive.mount("/content/drive")
        google_drive_path = "/content/drive/My Drive/Books/LLMs-From-Scratch/ch07/colab/gpt2-medium355M-sft.pth"  # 用户需要根据实际情况调整这个路径
        shutil.copy(google_drive_path, ".")
    else:
        print(
            f"Could not find '{finetuned_model_path}'.\n"
            "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
        )

# 导入GPT模型类
from previous_chapters import GPTModel

# 定义基础配置
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # dropout率
    "qkv_bias": True         # 查询-键-值偏置
}

# 定义不同规模的GPT模型配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 选择使用的GPT模型规模
CHOOSE_MODEL = "gpt2-medium (355M)"

# 更新基础配置为所选模型的配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 初始化GPT模型
model = GPTModel(BASE_CONFIG)

# 加载微调后的模型状态字典
model.load_state_dict(
    torch.load(
        "gpt2-medium355M-sft.pth",
        map_location=torch.device("cpu"),
        weights_only=True
    )
)
model.eval();  # 设置模型为评估模式

# 定义一个指令，要求将主动句转换为被动句
prompt = """Below is an instruction that describes a task. Write a response
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""

# 导入生成文本、文本到token ID、token ID到文本的函数
from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# 设置随机种子以保证结果可复现
torch.manual_seed(123)

# 生成文本
token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)

# 将token ID转换回文本
response = token_ids_to_text(token_ids, tokenizer)
print(response)

# 提取响应文本
def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

response = extract_response(response, prompt)
print(response)

# 初始化策略模型和参考模型
policy_model = model
reference_model = GPTModel(BASE_CONFIG)
reference_model.load_state_dict(
    torch.load(
        "gpt2-medium355M-sft.pth",
        map_location=torch.device("cpu"),
        weights_only=True
    )
)
reference_model.eval()

# 将模型移动到设备上（如GPU）
policy_model.to(device)
reference_model.to(device);

# 定义计算DPO损失的函数
import torch.nn.functional as F

def compute_dpo_loss(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """
    # 计算策略模型和参考模型的对数概率比率
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # 计算DPO损失
    losses = -F.logsigmoid(beta * logits)

    # 计算跟踪训练进度的可选值
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # 返回平均损失和奖励
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

# 定义计算对数概率的函数
def compute_logprobs(logits, labels, selection_mask=None):
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """
    # 将标签向前移动一位
    labels = labels[:, 1:].clone()

    # 截断logits以匹配标签的num_tokens
    logits = logits[:, :-1, :]

    # 计算log_softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # 收集实际标签的对数概率
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # 应用掩码以过滤掉填充token
        selected_log_probs = selected_log_probs * mask

        # 计算平均对数概率，不包括填充token
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)

# 示例数据
logits = torch.tensor(
    [[2.0, 1.0, 0.1],
     [0.5, 2.5, 0.3]])  # Shape: (2, 3)
targets = torch.tensor([0, 2])  # Shape: (2,)

# 使用torch.gather手动计算损失
log_softmax_logits = F.log_softmax(logits, dim=1)  # Shape: (2, 3)
selected_log_probs = torch.gather(
    input=log_softmax_logits,
    dim=1,
    index=targets.unsqueeze(1), # Shape 2, 1
).squeeze(1)  # Shape: (2,)
manual_loss = -selected_log_probs.mean()  # 批次平均

# PyTorch损失
cross_entropy_loss = F.cross_entropy(logits, targets)

# 打印手动损失和PyTorch损失
print(manual_loss, cross_entropy_loss)

# 示例：使用torch.gather
t = torch.tensor(
  [[1., 2.,],
   [3., 4.]]
)

m = torch.tensor(
  [[1, 1],
   [0, 1]]
)

torch.gather(input=t, dim=-1, index=m)


# 定义计算DPO损失的函数，用于处理一批数据
def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    #计算输入批次的DPO损失

    # 计算策略模型对选中响应的对数概率
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    # 计算策略模型对未选中响应的对数概率
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    # 计算参考模型对选中响应的对数概率
    ref_chosen_log_probas = compute_logprobs(
        logits=reference_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    # 计算参考模型对未选中响应的对数概率
    ref_rejected_log_probas = compute_logprobs(
        logits=reference_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    # 计算DPO损失和奖励
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    # 返回损失和奖励
    return loss, chosen_rewards, rejected_rewards

# 使用torch.no_grad()来关闭梯度计算，提高性能
with torch.no_grad():
    loss = compute_dpo_loss_batch(batch, policy_model, reference_model, beta=0.1)
# 打印损失
print(loss)

# 定义计算整个数据加载器上DPO损失的函数
def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    #对整个数据加载器应用compute_dpo_loss_batch

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    # 如果数据加载器为空，则返回NaN
    if len(data_loader) == 0:
        return float("nan")

    # 如果num_batches没有指定或者超过了数据加载器的批次数，则设置为数据加载器的批次数
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    # 遍历数据加载器中的批次
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            # 累积损失和奖励
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

        else:
            break

    # 计算平均损失和奖励
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    # 返回平均损失和奖励
    return total_loss, total_chosen_rewards, total_rejected_rewards

# 定义评估DPO损失的函数，用于训练和验证数据集
def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    #计算训练和验证数据集的DPO损失

    policy_model.eval()  # 设置策略模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    # 构建结果字典
    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()  # 设置策略模型为训练模式
    return res  # 返回结果

# 导入生成并打印样本的函数
from previous_chapters import generate_and_print_sample

# 定义简单的DPO训练模型函数
def train_model_dpo_simple(
    policy_model, reference_model, train_loader, val_loader,
    optimizer, num_epochs, beta,
    eval_freq, eval_iter, start_context, tokenizer
):
    # 初始化跟踪损失和token的字典
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        policy_model.train()  # 设置模型为训练模式

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()  # 重置梯度

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型权重

            tokens_seen += batch["chosen"].numel()  # 累积处理的token数量
            global_step += 1  # 累积步骤

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Epoch {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )

        # 每个epoch后打印样本文本
        generate_and_print_sample(
            model=policy_model,
            tokenizer=tokenizer,
            device=loss.device,
            start_context=start_context
        )

    return tracking  # 返回跟踪结果

# 设置随机种子以保证结果可复现性
torch.manual_seed(123)  # 由于数据加载器中的混洗

# 评估DPO损失
res = evaluate_dpo_loss_loader(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    beta=0.1,
    eval_iter=5
)

# 打印训练和验证损失
print("Training loss:", res["train_loss"])
print("Validation loss:", res["val_loss"])

# 打印训练和验证的奖励边际
print("Train reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
print("Val reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])

# 再次设置随机种子以保证结果可复现性
torch.manual_seed(123)


# 遍历验证数据集中的前三个条目
for entry in val_data[:3]:
    # 格式化输入文本
    input_text = format_input(entry)
    # 使用模型生成token IDs
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    # 将token IDs转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 提取响应文本
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    # 打印输入文本、正确响应和模型响应
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("\n-------------------------------------\n")

# 导入time模块，用于计算训练时间
import time

# 记录开始时间
start_time = time.time()

# 设置随机种子以保证结果可复现性
torch.manual_seed(123)

# 初始化AdamW优化器
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)

# 设置训练周期
num_epochs = 1
# 调用训练函数
tracking = train_model_dpo_simple(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    beta=0.1, # beta值在0.1和0.5之间
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[2]),
    tokenizer=tokenizer
)

# 记录结束时间并计算训练耗时
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入绘制损失函数
from previous_chapters import plot_losses

# 创建epoch的tensor
epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))
# 绘制损失曲线
plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=tracking["train_losses"],
    val_losses=tracking["val_losses"],
    label="loss"
)

# 计算训练和验证的奖励边际
train_reward_margins = [i-j for i,j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
val_reward_margins = [i-j for i,j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

# 绘制奖励边际曲线
plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=train_reward_margins,
    val_losses=val_reward_margins,
    label="loss"
)

# 再次设置随机种子以保证结果可复现性
torch.manual_seed(123)

# 遍历验证数据集中的前三个条目
for entry in val_data[:3]:
    # 格式化输入文本
    input_text = format_input(entry)
    # 使用参考模型生成token IDs
    token_ids = generate(
        model=reference_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    # 将token IDs转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 提取参考模型的响应文本
    reference_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    # 使用策略模型生成token IDs
    token_ids = generate(
        model=policy_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    # 将token IDs转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 提取策略模型的响应文本
    policy_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    # 打印输入文本、正确响应、参考模型响应和策略模型响应
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nReference model response:\n>> {reference_response_text.strip()}")
    print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
    print("\n-------------------------------------\n")

# 再次设置随机种子以保证结果可复现性
torch.manual_seed(123)

# 遍历测试数据集中的前三个条目
for entry in test_data[:3]:
    # 格式化输入文本
    input_text = format_input(entry)
    # 使用参考模型生成token IDs
    token_ids = generate(
        model=reference_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    # 将token IDs转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 提取参考模型的响应文本
    reference_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    # 使用策略模型生成token IDs
    token_ids = generate(
        model=policy_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    # 将token IDs转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 提取策略模型的响应文本
    policy_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    # 打印输入文本、正确响应、参考模型响应和策略模型响应
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nReference model response:\n>> {reference_response_text.strip()}")
    print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
    print("\n-------------------------------------\n")