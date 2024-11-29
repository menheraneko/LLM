from gpt import TransformerBlock
import torch



# 4.1
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
} # 参数展示

block = TransformerBlock(GPT_CONFIG_124M) # 创建块

total_params = sum(p.numel() for p in block.ff.parameters()) # 参数和
print(f"Total number of parameters in feed forward module: {total_params:,}") # 输出总参数

total_params = sum(p.numel() for p in block.att.parameters()) # 参数和
print(f"Total number of parameters in attention module: {total_params:,}") # 输出






# 4.2


def get_config(base_config, model_name="gpt2-small"): # 根据字符串获取不同模型参数
    GPT_CONFIG = base_config.copy() # copy

    if model_name == "gpt2-small": # small
        GPT_CONFIG["emb_dim"] = 768
        GPT_CONFIG["n_layers"] = 12
        GPT_CONFIG["n_heads"] = 12

    elif model_name == "gpt2-medium": # medium
        GPT_CONFIG["emb_dim"] = 1024
        GPT_CONFIG["n_layers"] = 24
        GPT_CONFIG["n_heads"] = 16

    elif model_name == "gpt2-large": # large
        GPT_CONFIG["emb_dim"] = 1280
        GPT_CONFIG["n_layers"] = 36
        GPT_CONFIG["n_heads"] = 20

    elif model_name == "gpt2-xl": # xl
        GPT_CONFIG["emb_dim"] = 1600
        GPT_CONFIG["n_layers"] = 48
        GPT_CONFIG["n_heads"] = 25

    else: # 错误
        raise ValueError(f"Incorrect model name {model_name}")

    return GPT_CONFIG


def calculate_size(model):  # 计算大小

    total_params = sum(p.numel() for p in model.parameters()) # 参数和
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters()) #参数和
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4 # 字节

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024) # mb

    print(f"Total size of the model: {total_size_mb:.2f} MB")

    from gpt import GPTModel

    for model_abbrev in ("small", "medium", "large", "xl"):
        model_name = f"gpt2-{model_abbrev}" # 名称
        CONFIG = get_config(GPT_CONFIG_124M, model_name=model_name) # 参数
        model = GPTModel(CONFIG) # 模型
        print(f"\n\n{model_name}:") # 输出
        calculate_size(model) # 计算模型大小



# 4.3
import torch.nn as nn
from gpt import MultiHeadAttention, LayerNorm, FeedForward


# 此处代码和ch04相同，不做解释
# 此处代码和ch04相同，不做解释
# 此处代码和ch04相同，不做解释
# 此处代码和ch04相同，不做解释

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_attn"], # NEW: dropout for multi-head attention
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


# 此处代码和ch04相同，不做解释
# 此处代码和ch04相同，不做解释
# 此处代码和ch04相同，不做解释
# 此处代码和ch04相同，不做解释
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"]) # NEW: dropout for embedding layers

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123) # 种子
model = GPTModel(GPT_CONFIG_124M) # 模型创建