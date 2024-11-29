import json
from openai import OpenAI

# Load API key from a JSON file.
# Make sure to replace "sk-..." with your actual API key from https://platform.openai.com/api-keys
import json  # 导入JSON库用于处理JSON数据
from openai import OpenAI  # 导入OpenAI库以与API交互
from tqdm import tqdm  # 导入tqdm用于显示进度条
from pathlib import Path  # 导入Path用于文件管理

# 从配置文件中加载API密钥
with open("config.json", "r") as config_file:
    config = json.load(config_file)  # 读取配置文件内容
    api_key = config["OPENAI_API_KEY"]  # 提取API密钥

# 创建OpenAI客户端实例
client = OpenAI(api_key=api_key)

# 函数：运行ChatGPT模型并返回响应
def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 温度设置为0以获取确定性的输出
        seed=123,  # 设置种子以确保一致性（如支持）
    )
    return response.choices[0].message.content  # 返回响应内容

# 示例提示以测试模型
prompt = "如果收到此消息，请回复'hello world'。"
print(run_chatgpt(prompt, client))  # 打印对测试提示的响应

# 加载评估数据集
json_file = "eval-example-data.json"
with open(json_file, "r") as file:
    json_data = json.load(file)  # 将JSON数据加载到Python对象中

# 打印数据集中条目的数量
print("条目数量:", len(json_data))

# 函数：格式化输入以供模型评估
def format_input(entry):
    instruction_text = (
        "以下是一个指令，描述了一个任务。请写出适当完成请求的响应。"  
        f"\n\n### Instruction:\n{entry['instruction']}"  # 添加指令文本
    )

    # 如果输入存在，则包括该输入
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text  # 返回组合后的指令和输入

# 对部分条目进行评估
for entry in json_data[:5]:
    prompt = (
        f"给定输入 `{format_input(entry)}` "  
        f"和正确输出 `{entry['output']}`， "  
        f"对模型响应 `{entry['model 1 response']}` "  
        "进行0到100的评分，其中100是最佳分数。"
    )
    print("\n数据集响应:")
    print(">>", entry['output'])  # 打印正确输出
    print("\n模型响应:")
    print(">>", entry["model 1 response"])  # 打印模型的响应
    print("\n分数:")
    print(">>", run_chatgpt(prompt, client))  # 获取并打印模型给出的分数
    print("\n-------------------------")  # 分隔符，方便阅读


# 函数：为数据集中的所有模型响应生成分数
def generate_model_scores(json_data, json_key, client):
    scores = []  # 存储分数的列表
    for entry in tqdm(json_data, desc="评分条目"):  # 为条目评分时显示进度条
        prompt = (
            f"给定输入 `{format_input(entry)}` "  
            f"和正确输出 `{entry['output']}`， "  
            f"对模型响应 `{entry[json_key]}` "  
            "进行0到100的评分，其中100是最佳分数。"  
            "仅回复数字。"
        )
        score = run_chatgpt(prompt, client)  # 从聊天模型请求评分
        try:
            scores.append(int(score))  # 将分数转换为整数并添加到列表中
        except ValueError:
            continue  # 如果转换失败，则跳过此条目

    return scores  # 返回分数列表

# 如果保存分数的目录不存在，则创建该目录
save_dir = Path("scores")
save_dir.mkdir(parents=True, exist_ok=True)

# 评估两个模型并存储它们的分数
for model in ("model 1 response", "model 2 response"):
    scores = generate_model_scores(json_data, model, client)  # 生成分数
    print(f"\n{model}")  # 打印当前模型名称
    print(f"评分数量: {len(scores)} / {len(json_data)}")  # 打印评分数量
    print(f"平均分: {sum(scores)/len(scores):.2f}\n")  # 打印平均分数

    # 可选：将分数保存到文件
    save_path = save_dir / f"gpt4-{model.replace(' ', '-')}.json"
    with open(save_path, "w") as file:
        json.dump(scores, file)  # 将分数写入JSON文件