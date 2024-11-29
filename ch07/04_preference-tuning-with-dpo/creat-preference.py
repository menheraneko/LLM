import urllib.request


import json
import urllib.request
from pathlib import Path
import random
from tqdm import tqdm


# 函数：查询模型以获得响应
def query_model(prompt, model="llama3.1:70b", url="http://localhost:11434/api/chat"):
    # 创建数据负载，作为字典格式
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt  # 用户提供的提示内容
            }
        ],
        "options": {
            "seed": 123,  # 设置随机种子
            "temperature": 0,  # 设置温度为0以提高响应的一致性
        }
    }

    # 将字典转换为JSON格式的字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建请求对象，设置方法为POST并添加必要的头部
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")  # 设置请求头为JSON格式

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取并解码响应
        while True:
            line = response.readline().decode("utf-8")  # 按行读取
            if not line:  # 如果没有更多行，退出循环
                break
            response_json = json.loads(line)  # 将行数据解析为JSON
            response_data += response_json["message"]["content"]  # 拼接响应内容

    return response_data  # 返回完整的响应数据


# 查询模型的示例
result = query_model("What do Llamas eat?")
print(result)  # 打印模型的响应

# 加载JSON数据文件
json_file = Path("..", "01_main-chapter-code", "instruction-data.json")
with open(json_file, "r") as file:
    json_data = json.load(file)  # 将JSON数据加载到Python对象中

print("Number of entries:", len(json_data))  # 打印数据条目的数量


# 函数：格式化输入以供模型评估
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"  # 添加指令
    )

    # 如果输入存在，则包含输入文本
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    # 返回组合后的指令和输入
    return instruction_text + input_text


# 使用随机选择生成更加有礼貌或不太有礼貌的响应
for entry in json_data[:5]:
    politeness = random.choice(["polite", "impolite"])  # 随机选择礼貌性
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"slightly rewrite the output to be more {politeness}. "
        "Keep the modification minimal. "
        "Only return the generated response and nothing else."
    )
    print("\nDataset response:")
    print(">>", entry['output'])  # 打印数据集中正确的响应
    print(f"\n{politeness} response:")
    print(">>", query_model(prompt))  # 打印模型生成的响应


# 函数：生成模型的响应
def generate_model_responses(json_data):
    for i, entry in enumerate(tqdm(json_data, desc="Writing entries")):  # 显示进度条
        politeness = random.choice(["polite", "impolite"])  # 随机选择礼貌性
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"slightly rewrite the output to be more {politeness}. "
            "Keep the modification minimal. "
            "Only return the generated response and nothing else."
        )
        response = query_model(prompt)  # 查询模型以获取响应

        # 根据礼貌性选择键值对
        if politeness == "polite":
            json_data[i]["chosen"] = response  # 记录生成的响应
            json_data[i]["rejected"] = entry["output"]  # 原响应被拒绝
        else:
            json_data[i]["rejected"] = response  # 生成的响应被拒绝
            json_data[i]["chosen"] = entry["output"]  # 原响应被接受


# 将更新后的数据写入JSON文件
with open("instruction-data-with-preference.json", "w") as file:
    json.dump(json_data, file, indent=4)  # 使用缩进将数据写入文件