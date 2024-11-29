import urllib.request
import json


import json  # 导入 JSON 库，用于处理 JSON 数据
import urllib.request  # 导入请求库，用于发送 HTTP 请求
from tqdm import tqdm  # 导入进度条库，用于显示进度条
from pathlib import Path  # 用于文件路径管理

# 定义查询模型的函数
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 创建数据有效载荷，这里使用字典形式
    data = {
        "model": model,  # 模型名称
        "messages": [  # 消息列表
            {
                "role": "user",  # 用户角色
                "content": prompt  # 用户输入的提示
            }
        ],
        "options": {  # 设置选项，用于确保响应的一致性
            "seed": 123,  # 随机种子
            "temperature": 0,  # 温度设置，0 表示确定性
            "num_ctx": 2048  # 上下文数量
        }
    }

    # 将字典转换为 JSON 格式的字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建请求对象，设置方法为 POST 并添加必要的头部
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")  # 指定内容类型为 JSON

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取和解码响应
        while True:
            line = response.readline().decode("utf-8")  # 逐行读取
            if not line:  # 如果没有更多行，退出循环
                break
            response_json = json.loads(line)  # 将 JSON 格式行解析为字典
            response_data += response_json["message"]["content"]  # 拼接响应内容

    return response_data  # 返回最终的响应内容


# 发送一个示例查询
result = query_model("What do Llamas eat?")
print(result)  # 打印响应结果

# 指定 JSON 文件名
json_file = "eval-example-data.json"

# 打开 JSON 文件并加载内容
with open(json_file, "r") as file:
    json_data = json.load(file)  # 将文件内容加载为 Python 对象

# 打印数据集中条目的数量
print("Number of entries:", len(json_data))

# 定义格式化输入的函数
def format_input(entry):
    # 构造指令文本，包括任务说明
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "  
        f"appropriately completes the request."  
        f"\n\n### Instruction:\n{entry['instruction']}"  # 添加任务指令
    )

    # 如果有输入，则格式化输入文本
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text  # 返回组合后的文本


# 遍历前 5 个条目进行评分
for entry in json_data[:5]:
    prompt = (f"Given the input `{format_input(entry)}` "  
              f"and correct output `{entry['output']}`, "  
              f"score the model response `{entry['model 1 response']}`"  
              f" on a scale from 0 to 100, where 100 is the best score. "
              )
    print("\nDataset response:")  # 打印数据集响应
    print(">>", entry['output'])
    print("\nModel response:")  # 打印模型响应
    print(">>", entry["model 1 response"])
    print("\nScore:")  # 打印评分
    print(">>", query_model(prompt))
    print("\n-------------------------")  # 分隔符


# 定义生成模型评分的函数
def generate_model_scores(json_data, json_key):
    scores = []  # 初始化评分列表
    for entry in tqdm(json_data, desc="Scoring entries"):  # 遍历 JSON 数据，并显示进度条
        prompt = (
            f"Given the input `{format_input(entry)}` "  
            f"and correct output `{entry['output']}`, "  
            f"score the model response `{entry[json_key]}`"  
            f" on a scale from 0 to 100, where 100 is the best score. "  
            f"Respond with the integer number only."
        )
        score = query_model(prompt)  # 调用查询模型获取评分
        try:
            scores.append(int(score))  # 将评分转换为整数并添加到列表中
        except ValueError:  # 如果转换失败，则跳过
            continue

    return scores  # 返回评分列表


# 逐个模型生成评分
for model in ("model 1 response", "model 2 response"):
    # 生成评分
    scores = generate_model_scores(json_data, model)
    print(f"\n{model}")  # 打印当前模型名称
    print(f"Number of scores: {len(scores)} of {len(json_data)}")  # 打印评分数量
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")  # 打印平均评分

    # 可选：保存评分到文件
    save_path = Path("scores") / f"llama3-8b-{model.replace(' ', '-')}.json"
    with open(save_path, "w") as file:
        json.dump(scores, file)  # 将评分写入 JSON 文件