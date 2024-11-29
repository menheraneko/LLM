from pathlib import Path
from previous_chapters import GPTModel
import tiktoken
import torch

from pathlib import Path

# 定义的路径
finetuned_model_path = Path("review_classifier.pth")

# 检查文件是否存在
if not finetuned_model_path.exists():
    print(
        f"Could not find '{finetuned_model_path}'.\n"  
        "Run the `ch06.ipynb` notebook to finetune and save the finetuned model."
    )

# 定义基本配置
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # Dropout率
    "qkv_bias": True         # 是否使用查询-键-值偏置
}

# 不同模型的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 选择要使用的模型
CHOOSE_MODEL = "gpt2-small (124M)"

# 更新
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 初始化
model = GPTModel(BASE_CONFIG)

# 将模型转换为分类器，设置输出层
num_classes = 2  # 设置分类数量（这里是二分类）
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# 加载预训练权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU
# 加载模型的状态权重，将其映射到设备上
model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
# 加载到模型
model.load_state_dict(model_state_dict)
# 评估模式
model.to(device)
model.eval()

# 获取tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 分类函数
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()  # 评估模式

    # 准备输入
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]  # 支持的上下文长度

    # 如果输入过长，则截断
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # 使用填充token填充序列到最大长度
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 添加batch维度

    with torch.no_grad():  # 不计算梯度
        logits = model(input_tensor.to(device))[:, -1, :]  # 获取最后输出token的logits
    predicted_label = torch.argmax(logits, dim=-1).item()  # 获取标签

    # 返回分类结果
    return "spam" if predicted_label == 1 else "not spam"

# 测试文本1
text_1 = (
    "You are a winner you have been specially"  
    " selected to receive $1000 cash or a $2000 award."
)

# 输出文本1的分类结果
print(classify_review(
    text_1, model, tokenizer, device, max_length=120
))

# 测试文本2
text_2 = (
    "Hey, just wanted to check if we're still on"  
    " for dinner tonight? Let me know!"
)

# 输出文本2的分类结果
print(classify_review(
    text_2, model, tokenizer, device, max_length=120
))