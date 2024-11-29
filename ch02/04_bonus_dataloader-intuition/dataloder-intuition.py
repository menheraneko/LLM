from importlib.metadata import version
import torch
from torch.utils.data import Dataset, DataLoader


with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):
        f.write(f"{number} ") # 写入文件，由到1001的数字构成

# 此处定义和main一样
# 此处定义和main一样
# 此处定义和main一样
# 此处定义和main一样
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Modification
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        token_ids = [int(i) for i in txt.strip().split()]

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() # 打开数字文本文件


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False) # 获取dataloader对象

data_iter = iter(dataloader) # 迭代器
first_batch = next(data_iter) # 第一批次
print(first_batch)

second_batch = next(data_iter) # 第二批次
print(second_batch)

third_batch = next(data_iter) # 第三批次
print(third_batch)

for batch in dataloader:
    pass # 找到最后一个批次

last_batch = batch # 最后一批次
print(last_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False) # 获取dataloader

for inputs, targets in dataloader: # 此处与上面同理
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)




torch.manual_seed(123) # 设置种子
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True) # 获取dataloader

for inputs, targets in dataloader: # 此处同理
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)