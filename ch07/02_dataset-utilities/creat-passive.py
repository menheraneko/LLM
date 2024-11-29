

import json
from openai import OpenAI
import json

# 读取配置文件以获取OpenAI API密钥
with open("config.json", "r") as config_file:
    config = json.load(config_file)  # 将配置文件内容加载为Python字典
    api_key = config["OPENAI_API_KEY"]  # 获取API密钥

# 初始化OpenAI客户端
client = OpenAI(api_key=api_key)

# 定义一个函数，用于与ChatGPT模型进行对话
def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    # 调用OpenAI的API生成聊天响应
    response = client.chat.completions.create(
        model=model,  # 指定使用的模型
        messages=[{"role": "user", "content": prompt}],  # 用户消息内容
        temperature=0.0,  # 设定温度参数为0，使得输出更稳定且确定
    )
    return response.choices[0].message.content  # 返回模型生成的内容

# 准备输入句子
sentence = "I ate breakfast"
# 构造提示，要求将句子转换为被动语态
prompt = f"Convert the following sentence to passive voice: '{sentence}'"
# 运行函数并打印结果
print(run_chatgpt(prompt, client))

# 要处理的JSON文件名称
json_file = "instruction-examples.json"

# 打开JSON文件并加载内容
with open(json_file, "r") as file:
    json_data = json.load(file)  # 将JSON文件内容加载为Python对象

# 打印条目的数量
print("Number of entries:", len(json_data))

# 遍历前5个条目进行处理
for entry in json_data[:5]:
    text = entry["output"]  # 获取当前条目的输出内容
    # 构造提示，要求将文本转换为被动语态
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"

    # 打印输入内容
    print("\nInput:")
    print(">>", text)  # 输出原始文本

    # 运行函数并打印转换后的输出
    print("\nOutput:")
    print(">>", run_chatgpt(prompt, client))  # 输出转换后的文本
    print("\n-------------------------")  # 分隔线


from tqdm import tqdm

# 使用进度条处理数据，并逐个给每个条目的output_2赋值
for i, entry in tqdm(enumerate(json_data[:5]), total=len(json_data[:5])):
    text = entry["output"]  # 获取当前条目的输出内容
    # 构造提示，要求将文本转换为被动语态
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
    json_data[i]["output_2"] = run_chatgpt(prompt, client)  # 将转换结果存储到output_2字段中

# 处理所有条目并保存被动语态结果
for i, entry in tqdm(enumerate(json_data), total=len(json_data)):
    text = entry["output"]  # 获取当前条目的输出内容
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
    json_data[i]["output_2"] = run_chatgpt(prompt, client)  # 将结果存储到output_2字段

# 创建新的JSON文件名，替换原始文件名后缀为-modified
new_json_file = json_file.replace(".json", "-modified.json")

# 将修改后的数据写入新的JSON文件中
with open(new_json_file, "w") as file:
    json.dump(json_data, file, indent=4)  # 将数据以缩进格式写入文件，便于阅读