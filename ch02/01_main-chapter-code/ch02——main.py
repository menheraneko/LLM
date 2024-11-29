import os
import urllib.request
import re

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

#本代码用于2.2main部分代码学习
if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt" #设置文本文件的路径
    urllib.request.urlretrieve(url, file_path) # 获取文本文件

with open("the-verdict.txt", "r", encoding="utf-8") as f: # 打开读入文件，编码使用utf-8
    raw_text = f.read() #读取文本文件中的内容

print("Total number of character:", len(raw_text)) # 输出长度
print(raw_text[:99]) # 输出第0到99列的内容，即0到99索引的内容。

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text) # 以空白\s划分text，作为tokenization的准备
print(result) # 输出划分后的result，这里为数组

result = re.split(r'([,.]|\s)', text) # 更细粒度的划分，增加，和。
print(result) # 此处将得到分词后的text，包括

result = [item for item in result if item.strip()] # 删除空字符串
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 处理更加复杂的句子，增加对其他符号的划分
result = [item.strip() for item in result if item.strip()] # 删除空字符串
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) # 对文本文件进行划分
preprocessed = [item.strip() for item in preprocessed if item.strip()] # 删除空串
print(preprocessed[:30]) # 输出前30个字符

print(len(preprocessed)) #输出总序列长度




#----------------------------------------------------------------------------
# 2.3部分代码
# 此处代码为学习代码的总结，为一个token处理类，具体如下:
class SimpleTokenizerV1:
    def __init__(self, vocab): # 传入字典以初始化
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()} # 所有键值对

    def encode(self, text): # 文本编码为token id
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 分词器，划分文本

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ] # 去空串
        ids = [self.str_to_int[s] for s in preprocessed] # 根据word查询到其在词表中对应的index，组成该文本的token id 返回
        return ids

    def decode(self, ids): # 解编码
        text = " ".join([self.int_to_str[i] for i in ids]) # 通过index查询到对应word token
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # 反划分
        return text




all_words = sorted(set(preprocessed)) # 利用set，把所有的token类型都区分开来，以建立简单词表
vocab_size = len(all_words) # 获取所有token种类的总数

print(vocab_size)
vocab = {token:integer for integer,token in enumerate(all_words)} # 以字典键值对方式存储词表 i: words

# 输出前50个词表元素
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


tokenizer = SimpleTokenizerV1(vocab) # 创建编解码器对象

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text) # token id编码
print(ids)
tokenizer.decode(ids) # 解码

tokenizer.decode(tokenizer.encode(text)) # 反向编解码





#----------------------------------------------------------------------------
# 2.4部分代码
# 此处代码为学习代码的总结，为一个token处理类，具体如下:
class SimpleTokenizerV2:
    def __init__(self, vocab): # 初始化不变
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text): # 编码中改进了遇见未知元素的处理

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed # 当遇见未知词，使用unk作为标记
        ]

        ids = [self.str_to_int[s] for s in preprocessed] # 返回token id
        return ids

    def decode(self, ids): # 无改进
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text




text = "Hello, world. Is this-- a test?"
#tokenizer.encode(text) # 编码，会导致错误，因为有不在词汇表中的词。

all_tokens = sorted(list(set(preprocessed))) # 获取token类型列表
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # 添加标记用于区分PAD和未在词表中的单词

vocab = {token:integer for integer,token in enumerate(all_tokens)} #重新建立字典词表
for i, item in enumerate(list(vocab.items())[-5:]): # 节约时间，只输出最后5列检验添加结果
    print(item)


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace." # 测试文本示例

text = " <|endoftext|> ".join((text1, text2)) # 添加文本分隔标记

print(text)
tokenizer.encode(text) #编码
tokenizer.decode(tokenizer.encode(text)) # 反向编解码





#----------------------------------------------------------------------------
# 2.5部分代码
tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
) # 测试文本示例

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # 加入特殊标记构建词表，并对text编码
print(integers)

strings = tokenizer.decode(integers) # 解码
print(strings)




#----------------------------------------------------------------------------
# 2.6部分代码
# dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride): # 初始化
        self.input_ids = []
        self.target_ids = []

        # Tokenize 文本，设置编码模式增加额外标记
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口把input分成max_length的各种序列，步进为stride
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length] # 当前处理的input
            target_chunk = token_ids[i + 1: i + max_length + 1] # 目标序列，模拟逐词推理
            self.input_ids.append(torch.tensor(input_chunk)) # 把张量加入input序列
            self.target_ids.append(torch.tensor(target_chunk)) # 加入target

    def __len__(self): # 输入长度
        return len(self.input_ids)

    def __getitem__(self, idx): # 输入和目标对
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0): # 创建dataloder，刚需传入文本

    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # 批大小
        shuffle=shuffle, # 随机打乱
        drop_last=drop_last, #是否丢弃batch的多余样本
        num_workers=num_workers # 子进程数量
    )

    return dataloader




with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() # 读取文本文件

enc_text = tokenizer.encode(raw_text) # 编码为token id
enc_sample = enc_text[50:] # 测试采样50个token
context_size = 4 # 设置窗口大小为4

x = enc_sample[:context_size] # 自变量，作为input
y = enc_sample[1:context_size+1] # 实际值，由于此处模拟逐词预测，所以序列位置比x多1，为下一个x位置

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1): # 遍历，输出实际值
    context = enc_sample[:i] # 当前上下文
    desired = enc_sample[i] # 要被预测的实际值

    print(context, "---->", desired) # 模拟预测的token id

for i in range(1, context_size+1): # 遍历，输出预测值
    context = enc_sample[:i] # 当前上下文
    desired = enc_sample[i] # 当前预测的值

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired])) # 模拟预测的token序列


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
) # 获取dataloder对象，每批次1，上下文长度4

data_iter = iter(dataloader) # 创建迭代器，便于取batch
first_batch = next(data_iter) # 获取第一个批次
print(first_batch)

second_batch = next(data_iter) #第二批次
print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False) # 批次大小改为8的测试

data_iter = iter(dataloader) # 迭代器
inputs, targets = next(data_iter) # 第一批次
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)








#----------------------------------------------------------------------------
# 2.7部分代码
input_ids = torch.tensor([2, 3, 5, 1]) # 取2，3，5，1的token id用于模拟input嵌入

vocab_size = 6 # 设置词表大小6
output_dim = 3 # 设置输出维度3

torch.manual_seed(123) # 设置初始化种子
embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # 创建嵌入层，input维度为词表大小，输出维度为3
print(embedding_layer.weight) # 输出层权重

print(embedding_layer(torch.tensor([3]))) # 嵌入测试
print(embedding_layer(input_ids)) # 嵌入测试





#----------------------------------------------------------------------------
# 2.8部分代码
vocab_size = 50257 # BytePair encoder的词表大小
output_dim = 256 # 输出维度

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # 创建embedding层

max_length = 4 # 设置最大长度4，控制窗口
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
) # 创建dataloader
data_iter = iter(dataloader) # 迭代器
inputs, targets = next(data_iter) # 第一批次数据

print("Token IDs:\n", inputs) # 相关信息
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs) #进行token嵌入
print(token_embeddings.shape) # 输出token vector大小

context_length = max_length # 上下文长度，即窗口大小
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) # 创建第二个嵌入层，用于嵌入位置向量

pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # 使用token的绝对位位置嵌入位置向量
print(pos_embeddings.shape) # 输出大小

input_embeddings = token_embeddings + pos_embeddings # 位置嵌入+token嵌入，获得input
print(input_embeddings.shape) # 输出大小，应当为（batch_size, feature_dim)






