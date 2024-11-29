import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 此处没有改动，和ch02中代码相同，不做解释
# 此处没有改动，和ch02中代码相同，不做解释
# 此处没有改动，和ch02中代码相同，不做解释

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

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

# 此处没有改动，和ch02中代码相同，不做解释
# 此处没有改动，和ch02中代码相同，不做解释
# 此处没有改动，和ch02中代码相同，不做解释
# 此处没有改动，和ch02中代码相同，不做解释
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


with open("small-text-sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer，使用gpt2模型
encoded_text = tokenizer.encode(raw_text) # 获取token id序列

vocab_size = 50257 # 词表长度
output_dim = 256 # 输出维度
max_len = 1024 # 最大长度
context_length = max_len # 上下文长度


token_embedding_layer = nn.Embedding(vocab_size, output_dim) # 创建对应token嵌入层
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) # 位置嵌入层

max_length = 4 # 最大长度
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length) # 获取dataloader

for batch in dataloader: # 逐批次处理
    x, y = batch

    token_embeddings = token_embedding_layer(x) # token嵌入
    pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # 位置嵌入

    input_embeddings = token_embeddings + pos_embeddings # 获取input

    break

print(input_embeddings.shape) # 形状


# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class CausalSelfAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, n_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec


# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)

torch.manual_seed(123) # 设置种子

context_length = max_length # 上下文长度
d_in = output_dim # 输出维度

num_heads=2 #头数
d_out = d_in // num_heads # 保证输出维度统一

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads) # 获取对象

batch = input_embeddings # 获取batch
context_vecs = mha(batch) # 计算上下文向量

print("context_vecs.shape:", context_vecs.shape) # 形状


# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
# 此处没有改动，和ch03中代码相同，不做解释
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec



torch.manual_seed(123) #设置种子

context_length = max_length # 上下文长度
d_in = output_dim # 输入维度
d_out = d_in # 输出维度

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2) # 获取对象

batch = input_embeddings # 获取batch
context_vecs = mha(batch) # 计算上下文向量

print("context_vecs.shape:", context_vecs.shape) #形状