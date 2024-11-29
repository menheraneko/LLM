from pathlib import Path

finetuned_model_path = Path("gpt2-medium355M-sft.pth") # 设置模型路径

if not finetuned_model_path.exists(): # 路径不存在
    print(
        f"Could not find '{finetuned_model_path}'.\n" # 输出错误信息
        "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."
    )

from previous_chapters import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
} # 基础参数

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
} # 不同模型参数

CHOOSE_MODEL = "gpt2-medium (355M)" # 选择模型名

BASE_CONFIG.update(model_configs[CHOOSE_MODEL]) # 更新参数

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")") # 分词，得到模型大小
model = GPTModel(BASE_CONFIG) # 获取模型

import torch

model.load_state_dict(torch.load(
    "gpt2-medium355M-sft.pth",
    map_location=torch.device("cpu"),
    weights_only=True
)) # 加载模型

model.eval(); # 评估模式

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") # tokenizer

prompt = """Below is an instruction that describes a task. Write a response 
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
""" # 设置prompt格式

from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip() # 替换response域标识，分离出回答数据

torch.manual_seed(123) # 种子

token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
) # 生成文本预测

response = token_ids_to_text(token_ids, tokenizer) # 解码回答
response = extract_response(response, prompt) # 解码，把回答部分单独分离
print(response) # 输出