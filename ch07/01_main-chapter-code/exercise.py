import tiktoken
from torch.utils.data import Dataset




# 7.1
def format_input(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    ) # 设置指令格式

    input_text = f"\n{entry['input']}" if entry["input"] else "" # 设置输入格式

    return instruction_text + input_text # 返回完成input



sample_data = [
    {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"},
    {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
] # 示例

print(format_input(sample_data[0])) # 输出示例的标准输出
print()
print(format_input(sample_data[1])) # 输出示例的标准输出

# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:

            ###################################################################
            # NEW: Use `format_input_phi` and adjust the response text template


            instruction_plus_input = format_input(entry) # 标准化指令数据，形成prompt
            response_text = f"\n<|assistant|>:\n{entry['output']}" # 设置回答格式



            ###################################################################
            full_text = instruction_plus_input + response_text # 完整数据
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            ) # 编码，并加入列表

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


tokenizer = tiktoken.get_encoding("gpt2")



# 7.2

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    ) # 设置指令格式

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else "" # input格式

    return instruction_text + input_text # 完整input


import torch
from torch.utils.data import Dataset


# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        ##########################################################################################
        # New: Separate list for instruction lengths

        self.instruction_lengths = [] # 初始化指令长度列表

        ##########################################################################################

        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            ##########################################################################################
            # New: collect instruction lengths


            instruction_length = len(tokenizer.encode(instruction_plus_input)) # 获取编码后的指令长度
            self.instruction_lengths.append(instruction_length) # 加入列表


            ##########################################################################################

    def __getitem__(self, index):
        return self.instruction_lengths[index], self.encoded_texts[index] # 返回指令长度和编码文本

    def __len__(self):
        return len(self.data)



import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")



# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
# 主体部分无变动，不做解释
def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for instruction_length, item in batch)  # New: batch is now a tuple

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:  # New: batch is now a tuple
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        ##########################################################################################

        targets[:instruction_length - 1] = -100  # 掩码所有input和指令token

        ##########################################################################################

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

sample_data = [
    {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."},
    {'instruction': 'Sort the following list in alphabetical order.', 'input': 'Zebra, Elephant, Crocodile', 'output': 'Crocodile, Elephant, Zebra'},
    {'instruction': 'Arrange the given numbers in descending order.', 'input': '5, 12, 8, 3, 15', 'output': '15, 12, 8, 5, 3.'}
] # 测试样例

from torch.utils.data import DataLoader

train_dataset = InstructionDataset(sample_data, tokenizer) # 创建dataset
train_loader = DataLoader(
    train_dataset,
    batch_size=len(sample_data),
    collate_fn=custom_collate_fn,
    num_workers=0
) # loader

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape) # 输出input和target的形状


print("Inputs:\n", inputs[1]) # 输出第一个示例
print("\n\nTargets:\n", targets[1]) # 第一个示例

print(tokenizer.decode(list(inputs[1]))) # 输出解码结果
non_masked_targets = targets[1][targets[1] != -100] # 没有掩码的部分

print(tokenizer.decode(list(non_masked_targets))) # 输出解码结果