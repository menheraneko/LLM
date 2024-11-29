import torch
import torch.nn as nn


# 此处主体没有改动，和ch03中代码相同，不做解释
# 此处主体没有改动，和ch03中代码相同，不做解释
# 此处主体没有改动，和ch03中代码相同，不做解释
# 此处主体没有改动，和ch03中代码相同，不做解释
class CausalAttentionWithoutBuffers(nn.Module): # 去除了保存缓冲中间矩阵的过程

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec





torch.manual_seed(123) # 种子

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
) # 测试样例

batch = torch.stack((inputs, inputs), dim=0) # 获取第一batch
context_length = batch.shape[1] # 上下文长度
d_in = inputs.shape[1] # 输入维度
d_out = 2 # 输出维度

ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0) # 获取对象

with torch.no_grad(): # 不使用梯度
    context_vecs = ca_without_buffer(batch) # 获取结果

print(context_vecs) # 输出

print("Machine has GPU:", torch.cuda.is_available())

#batch = batch.to("cuda")
#ca_without_buffer.to("cuda"); # 模型转移

with torch.no_grad():
    context_vecs = ca_without_buffer(batch) # 没有保存缓冲，所以此时输出结果会不同。因为矩阵没有随模型一起转移

print(context_vecs)

print("W_query.device:", ca_without_buffer.W_query.weight.device) # 位置检查
print("mask.device:", ca_without_buffer.mask.device) # 位置检查





import torch.nn as nn

# 此处主体没有改动，和ch03中代码相同，不做解释
# 此处主体没有改动，和ch03中代码相同，不做解释
# 此处主体没有改动，和ch03中代码相同，不做解释
# 此处主体没有改动，和ch03中代码相同，不做解释
class CausalAttentionWithBuffer(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Old:
        # self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

        # New:
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0) # 创建对象
#ca_with_buffer.to("cuda") # 转移

print("W_query.device:", ca_with_buffer.W_query.weight.device) # 位置检查
print("mask.device:", ca_with_buffer.mask.device)

with torch.no_grad():
    context_vecs = ca_with_buffer(batch) # 获得结果

print(context_vecs) # 输出

ca_without_buffer.state_dict() # 状态检查
ca_with_buffer.state_dict()

ca_with_buffer.mask[ca_with_buffer.mask == 1.] = 2.
ca_with_buffer.mask

torch.save(ca_with_buffer.state_dict(), "model.pth") # 保存

new_ca_with_buffer = CausalAttentionWithBuffer(d_in, d_out, context_length, 0.0) # 创建对像
new_ca_with_buffer.load_state_dict(torch.load("model.pth")) # 加载

new_ca_with_buffer.mask

ca_without_buffer.mask[ca_without_buffer.mask == 1.] = 2.

torch.save(ca_without_buffer.state_dict(), "model.pth") # 保存

new_ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0) # 对象
new_ca_without_buffer.load_state_dict(torch.load("model.pth")) # 加载

new_ca_without_buffer.mask