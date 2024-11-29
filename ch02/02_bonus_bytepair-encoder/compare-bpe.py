import tiktoken
from bpe_openai_gpt2 import get_encoder, download_vocab
import transformers
from transformers import GPT2Tokenizer


tik_tokenizer = tiktoken.get_encoding("gpt2") # 获取tokenizer

text = "Hello, world. Is this-- a test?" # 测试文本

integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # 编码

print(integers)
strings = tik_tokenizer.decode(integers) # 解码

print(strings)
print(tik_tokenizer.n_vocab)




download_vocab() # 下载
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".") # 从gpt2获取tokenizer
integers = orig_tokenizer.encode(text) # 编码

print(integers)
strings = orig_tokenizer.decode(integers) # 解码

print(strings)







hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # 获取预训练模型的tokenizer
hf_tokenizer(strings)["input_ids"] # input id




with open('../01_main-chapter-code/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read() # 读取文本

 # 测试运行时间，在Jupyter中执行
#%timeit orig_tokenizer.encode(raw_text)

#%timeit tik_tokenizer.encode(raw_text)

#%timeit hf_tokenizer(raw_text)["input_ids"]

#%timeit hf_tokenizer(raw_text, max_length=5145, truncation=True)["input_ids"]