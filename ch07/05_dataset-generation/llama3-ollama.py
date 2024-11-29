import urllib.request
import json

# 定义一个函数，用于向模型发送查询
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat", role="user"):
    # 创建数据负载，格式为字典
    data = {
        "model": model,
        "seed": 123,        # 为了得到确定性响应
        "temperature": 1.,   # 为了得到确定性响应
        "top_p": 1,
        "messages": [
            {"role": role, "content": prompt}
        ]
    }

    # 将字典转换为JSON格式的字符串，并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建一个请求对象，设置方法为POST，并添加必要的头部
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取和解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    # 返回响应数据
    return response_data

# 使用query_model函数查询“Llamas吃什么？”
result = query_model("What do Llamas eat?")
print(result)

# 定义一个函数，用于从文本中提取指令
def extract_instruction(text):
    for content in text.split("\n"):
        if content:
            return content.strip()

# 定义查询字符串
query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

# 使用query_model函数查询，并指定角色为助理
result = query_model(query, role="assistant")
# 提取指令
instruction = extract_instruction(result)
print(instruction)

# 使用提取的指令查询模型，角色为用户
response = query_model(instruction, role="user")
print(response)

# 导入tqdm模块，用于显示进度条
from tqdm import tqdm

# 设置数据集大小
dataset_size = 5
# 初始化数据集列表
dataset = []

# 使用进度条遍历数据集大小次数
for i in tqdm(range(dataset_size)):
    # 使用query_model函数查询，并指定角色为助理
    result = query_model(query, role="assistant")
    # 提取指令
    instruction = extract_instruction(result)
    # 使用提取的指令查询模型，角色为用户
    response = query_model(instruction, role="user")
    # 构建数据集条目
    entry = {
        "instruction": instruction,
        "output": response
    }
    # 将条目添加到数据集中
    dataset.append(entry)

# 将数据集保存为JSON文件
with open("instruction-data-llama3-7b.json", "w") as file:
    json.dump(dataset, file, indent=4)