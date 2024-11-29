import json

# 打开并加载 JSON 文件，获取 GPT-4 模型的响应
with open("gpt4-model-1-response.json", "r") as file:
    gpt4_model_1 = json.load(file)  # 将文件内容加载为 Python 对象

# 打开并加载 JSON 文件，获取 Llama 3 8B 模型的响应
with open("llama3-8b-model-1-response.json", "r") as file:
    llama3_8b_model_1 = json.load(file)  # 将文件内容加载为 Python 对象

# 导入所需的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化

# 将 GPT-4 和 Llama 3 8B 的响应存储到列表中
list1, list2 = gpt4_model_1, llama3_8b_model_1

# 创建散点图，显示两个模型的响应关系
plt.scatter(list1, list2)  # 绘制散点图，x轴为 GPT-4 响应，y轴为 Llama 3 8B 响应

# 绘制回归线
plt.plot(
    np.unique(list1),  # 使用 list1 的唯一值作为 x 轴
    np.poly1d(np.polyfit(list1, list2, 1))(np.unique(list1))  # 计算并绘制线性拟合的y值
)
plt.xlabel("GPT-4")  # 设置 x 轴标签
plt.ylabel("Llama3 8B")  # 设置 y 轴标签
plt.show()  # 显示图形

# 导入 pandas 和统计相关性计算库
import pandas as pd  # 用于数据处理
from scipy.stats import spearmanr, kendalltau  # 用于计算 Spearman 和 Kendall Tau 相关系数

# 计算皮尔逊相关系数
pearson_correlation = np.corrcoef(list1, list2)[0, 1]  # 计算 list1 和 list2 的皮尔逊相关系数

# 计算斯皮尔曼相关系数和 p 值
spearman_correlation, _ = spearmanr(list1, list2)  # 计算斯皮尔曼相关系数

# 计算肯德尔相关系数和 p 值
kendall_tau_correlation, _ = kendalltau(list1, list2)  # 计算肯德尔 Tau 相关系数

# 创建一个数据表，包含三种不同的相关系数
correlation_table = pd.DataFrame({
    "Pearson": [pearson_correlation],  # 添加皮尔逊相关系数
    "Spearman": [spearman_correlation],  # 添加斯皮尔曼相关系数
    "Kendall Tau": [kendall_tau_correlation]  # 添加肯德尔 Tau 相关系数
}, index=['Results'])  # 设置行索引为 'Results'

# 输出相关系数表
print(correlation_table)  # 打印相关性表格

