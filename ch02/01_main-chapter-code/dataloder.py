import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 此部分和main中定义并无差别，主要解释非定义部分的代码
# 此部分和main中定义并无差别，主要解释非定义部分的代码
# 此部分和main中定义并无差别，主要解释非定义部分的代码
# 此部分和main中定义并无差别，主要解释非定义部分的代码
# 此部分和main中定义并无差别，主要解释非定义部分的代码

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


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f: # 打开文本文件，准备导入数据
    raw_text = f.read() # 读取文本数据

tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer，使用gpt2的库。
encoded_text = tokenizer.encode(raw_text) # 进行token编码，获取token id 序列

vocab_size = 50257 # 设置词表大小
output_dim = 256 # 输出维度
context_length = 1024 # 上下文长度


token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # token嵌入层
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) # 位置嵌入层

max_length = 4 # 窗口大小，max_length
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length) # 获取dataloader对象
for batch in dataloader: # 逐批次处理
    x, y = batch # 批次数据

    token_embeddings = token_embedding_layer(x) # 嵌入token
    pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # 绝对位置嵌入

    input_embeddings = token_embeddings + pos_embeddings # 得到最终input vector

    break
print(input_embeddings.shape) # 输出形状，应当为[8, 4, 256]