import json
from openai import OpenAI

# Load API key from a JSON file.
# Make sure to replace "sk-..." with your actual API key from https://platform.openai.com/api-keys
# 打开配置文件并加载JSON数据
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    # 提取OpenAI的API密钥
    api_key = config["OPENAI_API_KEY"]

# 初始化OpenAI客户端
client = OpenAI(api_key=api_key)

# 定义一个函数，用于运行ChatGPT模型并获取响应
def run_chatgpt(prompt, client, model="gpt-4o-mini", system_prompt=None):
    # 如果提供了system_prompt，则添加到消息列表
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # 添加用户提示到消息列表
    messages.append({"role": "user", "content": prompt})
    # 调用API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        seed=123,
    )
    # 返回模型的响应内容
    return response.choices[0].message.content

# 定义一个提示并运行ChatGPT
prompt = f"Respond with 'hello world' if you got this message."
run_chatgpt(prompt, client)

# 导入Pathlib模块
from pathlib import Path

# 定义JSON文件路径
json_file = Path("..") / "01_main-chapter-code" / "instruction-data.json"

# 打开JSON文件并加载数据
with open(json_file, "r") as file:
    json_data = json.load(file)

# 打印条目数量
print("Number of entries:", len(json_data))

# 导入pprint模块
from pprint import pp as pprint

# 打印第一个条目的详细信息
pprint(json_data[0])

# 定义一个函数，用于生成指令提示
def instr_prompt_no_input(ins, outp):
    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of a given instruction."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of a given instruction. \n" + \
                "1. Why this instruction is not good? First analyse the instruction based on Complexity of the Topic, Level of Detail Required, Knowledge Required, Ambiguity of the Instruction and Logical Reasoning or Problem-Solving Involved. \n" + \
                "Then analyse why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n" + \
                "Finally analyse why this bad instruction lead to a bad answer. " +\
                "2. Based on the reason you provided, generate a new and complete instruction which is complex and difficult to answer directly. " + \
                "Make sure the new instruction is relevent but independent to the original instruction, which can be answered without knowing the original instruction, put the new instruction in the format of [New Instruction] your instruction [End]" +\
                "3. Answer the newly generated instruction as detailed as possible, in the format of [New Answer] your answer [End] \n"
    prompt = prompt_template.format(
        ins=ins, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt

# 定义一个条目并生成系统提示和用户提示
entry = json_data[2]
system_prompt, prompt = instr_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)

# 打印输出
print(output)

# 导入正则表达式模块
import re

# 定义一个函数，用于从文本中提取新指令
def extract_ins(text, no_input=True):
    if '[New Instruction]' in text:
        pattern = r'(\[New Instruction\])(.*?)(\[End\]|\[New Answer\]|New Answer:)'
    else:
        pattern = r'(New Instruction:)(.*?)(\[End\]|\[New Answer\]|New Answer:)'
    segments = re.findall(pattern, text, re.DOTALL)
    if len(segments) == 0:
        seg_ins = ''
    else:
        seg_ins = segments[0][1].strip()
    if seg_ins.endswith("\n\n3."):
        seg_ins = seg_ins[:-4]
    return seg_ins

# 定义一个函数，用于从文本中提取新输出
def extract_oup(text, no_input=True):
    if '[New Answer]' in text:
        pattern = r'(\[New Answer\])(.*?)(\[End\]|$)'
    else:
        pattern = r'(New Answer:)(.*?)(\[End\]|$)'
    segments = re.findall(pattern, text, re.DOTALL)
    if len(segments) == 0:
        seg_oup = ''
    else:
        seg_oup = segments[0][1].strip()
    return seg_oup

# 定义一个函数，用于从文本中提取指令和输出
def extract_instruction(text):
    if text == '':
        return []
    seg_ins = extract_ins(text, no_input=True)
    seg_oup = extract_oup(text, no_input=True)
    return [seg_ins, seg_oup]

# 从输出中提取新指令和新输出
new_instr, new_outp = extract_instruction(output)

# 打印新指令和新输出
print(new_instr)
print(new_outp)

# 定义一个函数，用于生成响应生成提示
def res_gen_prompt_no_input(ins, outp):
    sys_prompt = "You are a helpful, precise but picky assistant for checking the quality of the answer to a given instruction."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the answer to the given instruction. \n" + \
                "1. Why this answer is not good for the given instruction? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n" + \
                "2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible, in the format of [Better Answer] your answer [End] \n"
    prompt = prompt_template.format(
        ins=ins, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt

# 定义一个函数，用于生成响应生成提示（包含输入）
def res_gen_prompt_input(ins, inp, outp):
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer to a given instruction and its input."
    prompt_template = "[Instruction]\n{ins}\n\n[The Start of Input]\n{inp}\n\n[The End of Input]\n\n[The Start of Answer]\n{outp}\n\n[The End of Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like you to answer several questions related to the quality of the answer to the given instruction and corresponding input. \n" + \
                "1. Why this answer is not good for the given instruction and corresponding input? Analyse based on the Helpfulness, Relevance, Accuracy and Level of Details. \n" + \
                "2. Based on the reason you provided, generate a better answer, new and complete, as detailed as possible, in the format of [Better Answer] your answer [End] \n"
    prompt = prompt_template.format(
        ins=ins, inp=inp, outp=outp, criteria=criteria
    )
    return sys_prompt, prompt

# 定义一个条目并生成系统提示和用户提示
entry = json_data[2]
system_prompt, prompt = res_gen_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)

# 打印输出
print(output)

# 定义一个函数，用于从文本中提取响应

def extract_response(text):
    if text.count('[Better Answer]') >= 2:
        pattern = r'\[(Better Answer)\](.*?)(\[End\]|\[Better Answer\]|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    else:
        pattern = r'\[(Better Answer)\](.*?)(\[End\]|End|$)'
        segments = re.findall(pattern, text, re.DOTALL)
    return [segment[1].strip() for segment in segments]

# 从输出中提取响应
response = extract_response(output)[0]
print(response)

# 定义要处理的数据
data_to_process = json_data[:3]


# 导入tqdm模块，用于显示进度条
from tqdm import tqdm

# 定义一个函数，用于反射（改进）指令
def reflect_instructions(json_data, client):
    new_json_data = []
    # 遍历JSON数据中的每个条目，并显示进度条
    for entry in tqdm(json_data):
        # 如果条目中没有输入（input），则生成新的指令和输出
        if not entry["input"]:
            system_prompt, prompt = instr_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
            output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)
            new_instr, new_outp = extract_instruction(output)
            new_entry = {"instruction": new_instr, "input": "", "output": new_outp}
            new_json_data.append(new_entry)
        else:
            # 如果条目中有输入（input），则直接将条目添加到新数据中
            new_json_data.append(entry)
    # 返回新数据
    return new_json_data

# 定义要处理的数据
data_to_process = json_data[:3]

# 调用reflect_instructions函数，改进指令
new_json_data = reflect_instructions(data_to_process, client)

# 打印新数据中的前三个条目
for i in new_json_data[:3]:
    pprint(i)
    print("\n\n")

# 将新数据保存为JSON文件
with open("instruction-reflected.json", "w") as file:
    json.dump(new_json_data, file, indent=4)

# 定义要处理的数据
data_to_process = json_data[:3]

# 定义一个函数，用于反射（改进）响应
def reflect_responses(json_data, client):
    new_json_data = []
    # 遍历JSON数据中的每个条目，并显示进度条
    for entry in tqdm(json_data):
        # 如果条目中没有输入（input），则生成新的响应
        if not entry["input"]:
            system_prompt, prompt = res_gen_prompt_no_input(ins=entry["instruction"], outp=entry["output"])
            output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)
            new_response = extract_response(output)
            # 如果没有新的响应，则使用原始输出
            if not len(new_response):
                new_response = entry["output"]
            new_entry = {"instruction": entry["instruction"], "input": "", "output": new_response[0]}
            new_json_data.append(new_entry)
        else:
            # 如果条目中有输入（input），则生成新的响应
            system_prompt, prompt = res_gen_prompt_input(ins=entry["instruction"], inp=entry["input"],
                                                         outp=entry["output"])
            output = run_chatgpt(prompt=prompt, client=client, system_prompt=system_prompt)
            new_response = extract_response(output)
            # 如果没有新的响应，则使用原始输出
            if not len(new_response):
                new_response = entry["output"]
            new_entry = {"instruction": entry["instruction"], "input": entry["input"], "output": new_response[0]}
            new_json_data.append(new_entry)
    # 返回新数据
    return new_json_data

# 调用reflect_responses函数，改进响应
new_json_data = reflect_responses(data_to_process, client)

# 打印新数据中的前三个条目
for i in new_json_data[:3]:
    pprint(i)
    print("\n\n")