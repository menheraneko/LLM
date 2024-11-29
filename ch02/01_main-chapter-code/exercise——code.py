import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader



# 第一部分
tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer

integers = tokenizer.encode("Akwirw ier") # 测试编码
print(integers)

for i in integers: # 逐字母处理
    print(f"{i} -> {tokenizer.decode([i])}") # 输出token id，token对

# 逐个编码
tokenizer.encode("Ak")
tokenizer.encode("w")
tokenizer.encode("ir")
tokenizer.encode("w")
tokenizer.encode(" ")
tokenizer.encode("ier")
tokenizer.decode([33901, 86, 343, 86, 220, 959]) # 对指定token id 解码







# 第二部分

# 此部分定义和main无差别
# 此部分定义和main无差别
# 此部分定义和main无差别
# 此部分定义和main无差别

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

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


def create_dataloader(txt, batch_size=4, max_length=256, stride=128):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer
encoded_text = tokenizer.encode(raw_text) # 编码

vocab_size = 50257 # 词表大小
output_dim = 256 # 输出维度
max_len = 4 # 最大长度
context_length = max_len # 上下文长度

token_embedding_layer = torch.nn.Embedding(context_length, output_dim) # token嵌入层
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # 位置嵌入层

dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2) # 创建dataloader

for batch in dataloader: # 批次处理
    x, y = batch
    break