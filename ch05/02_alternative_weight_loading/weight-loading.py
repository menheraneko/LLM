from transformers import GPT2Model

# 定义可用的GPT模型名称与对应的预训练模型路径
model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",  # GPT-2 小模型 (124M 参数)
    "gpt2-medium (355M)": "openai-community/gpt2-medium",  # GPT-2 中型模型 (355M 参数)
    "gpt2-large (774M)": "openai-community/gpt2-large",  # GPT-2 大模型 (774M 参数)
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"  # GPT-2 超大模型 (1558M 参数)
}

# 选择使用的GPT模型名称
CHOOSE_MODEL = "gpt2-small (124M)"

# 从指定路径加载预训练模型并将其设置为评估模式
gpt_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="checkpoints")
gpt_hf.eval()

# 定义模型基础配置，包括词汇大小、上下文长度、丢弃率等
BASE_CONFIG = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,  # 丢弃率
    "qkv_bias": True  # 是否使用查询-键-值偏置
}

# 根据不同的模型配置设置对应的参数
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小模型配置
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 中型模型配置
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 大模型配置
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},  # 超大模型配置
}

# 更新基础配置为选择的模型配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


# 定义一个函数来检查两个张量的形状是否一致，若不一致则抛出错误
def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


# 导入numpy库
import numpy as np


# 定义加载权重的函数，将预训练模型的权重赋值给自定义模型
def load_weights(gpt, gpt_hf):
    # 获取预训练模型的状态字典（所有权重）
    d = gpt_hf.state_dict()

    # 将位置嵌入权重从预训练模型赋值给自定义模型
    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    # 将词嵌入权重从预训练模型赋值给自定义模型
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])

    # 遍历每一层的注意力机制（多头注意力）
    for b in range(BASE_CONFIG["n_layers"]):
        # 将注意力权重（Q, K, V）分割并赋值给自定义模型
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign_check(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign_check(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign_check(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 将注意力偏置（Q, K, V）分割并赋值给自定义模型
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign_check(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign_check(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign_check(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 将注意力投影层的权重和偏置赋值
        gpt.trf_blocks[b].att.out_proj.weight = assign_check(gpt.trf_blocks[b].att.out_proj.weight,
                                                             d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign_check(gpt.trf_blocks[b].att.out_proj.bias,
                                                           d[f"h.{b}.attn.c_proj.bias"])

        # 将前馈层的权重和偏置赋值
        gpt.trf_blocks[b].ff.layers[0].weight = assign_check(gpt.trf_blocks[b].ff.layers[0].weight,
                                                             d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign_check(gpt.trf_blocks[b].ff.layers[0].bias,
                                                           d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign_check(gpt.trf_blocks[b].ff.layers[2].weight,
                                                             d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign_check(gpt.trf_blocks[b].ff.layers[2].bias,
                                                           d[f"h.{b}.mlp.c_proj.bias"])

        # 将规范化层的缩放因子和偏置赋值
        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])


# 导入torch库
import torch
# 从之前的章节导入GPT模型定义
from previous_chapters import GPTModel

# 初始化自定义模型
gpt = GPTModel(BASE_CONFIG)

# 设置设备为GPU（如果可用），否则为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练模型的权重
load_weights(gpt, gpt_hf)

# 导入tiktoken库（用于文本编码）
import tiktoken
# 从之前的章节导入生成文本的函数和工具
from previous_chapters import generate, text_to_token_ids, token_ids_to_text

# 设置随机种子以保证结果的可重现性
torch.manual_seed(123)

# 获取GPT-2的tokenizer编码器
tokenizer = tiktoken.get_encoding("gpt2")

# 使用自定义模型生成文本
token_ids = generate(
    model=gpt.to(device),  # 将模型加载到设备上（GPU/CPU）
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),  # 将输入文本转换为token ID
    max_new_tokens=30,  # 生成的最大token数量
    context_size=BASE_CONFIG["context_length"],  # 上下文大小
    top_k=1,  # top-k采样策略
    temperature=1.0  # 温度参数，控制生成文本的随机性
)

# 打印生成的文本
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))