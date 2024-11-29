# 导入垃圾回收模块gc，用于手动管理内存清理
import gc
# 导入时间模块time，用于控制时间延迟
import time
# 导入PyTorch库torch，用于深度学习和GPU操作
import torch


def start_memory_tracking():
    """Initialize GPU memory tracking."""
    # 如果系统支持CUDA（即有NVIDIA GPU并安装了CUDA），则进行GPU内存追踪初始化
    if torch.cuda.is_available():
        # 重置并记录GPU内存分配的峰值状态
        torch.cuda.reset_peak_memory_stats()
    else:
        # 如果没有CUDA设备可用，打印警告信息
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")

def print_memory_usage():
    """Print maximum GPU memory allocated so far."""
    # 获取当前CUDA设备的最大内存分配量，并转换为GB
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    # 输出最大内存分配量（GB）
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")

def cleanup():
    """Free memory and reset memory stats."""
    # 执行垃圾回收操作，尝试清理不再使用的内存
    gc.collect()
    # 清空GPU缓存，释放已分配但未使用的内存
    torch.cuda.empty_cache()
    # 延迟3秒，以确保GPU缓存清空操作完成
    time.sleep(3)  # some buffer time to allow memory to clear
    # 重置CUDA设备的内存峰值统计数据
    torch.cuda.reset_peak_memory_stats()
    # 获取清理后GPU设备的最大内存分配，并转换为GB
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    # 输出清理后的最大内存分配量（GB）
    print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")














# 从之前的章节中导入GPTModel类，假设它是用于构建和训练GPT模型的基础类
from previous_chapters import GPTModel

# 定义基础配置字典BASE_CONFIG，包含一些GPT模型的通用参数
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表的大小，指定模型可以处理的不同单词或标记的数量
    "context_length": 1024,  # 上下文长度，指模型在进行预测时能够看到的最大token数（如：文本片段的长度）
    "drop_rate": 0.0,        # Dropout率，表示训练过程中随机丢弃神经网络神经元的比例，0.0表示没有dropout
    "qkv_bias": True         # 查询-键-值（QKV）偏置，是否在自注意力机制中使用QKV的偏置
}

# 定义一个字典model_configs，包含不同大小的GPT-2模型的具体配置参数
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 124M参数量的GPT-2配置，768维词嵌入，12层transformer，12个头的自注意力
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 355M参数量的GPT-2配置，1024维词嵌入，24层transformer，16个头的自注意力
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},   # 774M参数量的GPT-2配置，1280维词嵌入，36层transformer，20个头的自注意力
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},     # 1558M参数量的GPT-2配置，1600维词嵌入，48层transformer，25个头的自注意力
}

# 选择使用的模型大小，设置为 "gpt2-xl (1558M)"
CHOOSE_MODEL = "gpt2-xl (1558M)"

# 使用update方法将选择的模型的特定配置参数合并到基础配置BASE_CONFIG中
# model_configs[CHOOSE_MODEL] 提供了选择模型的具体配置（如emb_dim, n_layers, n_heads）
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


# 启动内存追踪
start_memory_tracking()

# 创建一个基于基本配置的GPT模型实例
model = GPTModel(BASE_CONFIG)

# 设置计算设备为CUDA（GPU），如果有GPU可用
device = torch.device("cuda")

# 将模型移动到GPU设备上
model.to(device)

# 打印当前内存使用情况
print_memory_usage()

# 测试模型是否能正常工作（无需追踪内存）
# 创建一个输入张量并将其转移到GPU上
test_input = torch.tensor([[1, 2, 3]]).to(device)

# 将模型设置为评估模式，这会关闭诸如dropout等训练时特有的行为
model.eval()

# 在不计算梯度的情况下进行前向传播（即推理）
with torch.no_grad():
    # 将测试输入传入模型进行计算
    model(test_input)

# 训练代码将在这里添加

# 将模型设置为训练模式，这会启用训练时的行为（例如dropout）
model.train()

# 将模型的权重保存到文件中，文件名为 "model.pth"
torch.save(model.state_dict(), "model.pth")

# 删除模型和测试输入以释放内存
del model, test_input

# 执行清理操作，释放任何可能被占用的资源
cleanup()














# 然后加载预训练的权重

# 开始追踪内存使用情况，通常用于检测内存使用变化，帮助优化内存管理
start_memory_tracking()

# 使用基础配置创建一个新的GPT模型实例
model = GPTModel(BASE_CONFIG)

# 将模型移到指定的设备（如GPU或者CPU），确保模型能在正确的硬件上运行
model.to(device)

# 加载保存的预训练权重，`torch.load("model.pth", map_location=device)` 会根据 `device` 加载模型
# `weights_only=True` 表示只加载模型权重（不加载其他信息，比如优化器状态等）
model.load_state_dict(
    torch.load("model.pth", map_location=device, weights_only=True)
)

# 再次确保模型在指定的设备上运行
model.to(device)

# 设置模型为评估模式。评估模式下，像 dropout 或 batch normalization 等操作会变得不同，适合推理阶段
model.eval()

# 打印内存使用情况，显示内存的分配情况
print_memory_usage()

# 测试模型是否能够正常工作（这里不需要追踪内存使用）
# 创建一个简单的测试输入张量，并将其移到指定的设备上
test_input = torch.tensor([[1, 2, 3]]).to(device)

# 设置模型为评估模式，虽然已经设置过一次，但在这里再次确保其处于评估状态
model.eval()

# 在没有梯度计算的情况下执行推理（使用`torch.no_grad()`禁用梯度计算，节省内存）
with torch.no_grad():
    model(test_input)  # 传入测试输入，执行前向传播

# 删除模型和测试输入，释放内存
del model, test_input

# 调用`cleanup()`，进行其他资源清理（例如释放缓存或进行特定的内存管理）
cleanup()













# 开始内存追踪，通常用于检查内存的使用情况
start_memory_tracking()

# 创建一个GPTModel实例，使用BASE_CONFIG配置，并将模型移动到指定的设备（例如GPU或CPU）
model = GPTModel(BASE_CONFIG).to(device)

# 从本地路径加载模型权重文件（model.pth），并将其映射到CPU内存中
state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

# 打印当前内存使用情况
print_memory_usage()

# 不计算梯度的情况下，逐一将加载的权重（state_dict）复制到模型的参数中
with torch.no_grad():
    # 遍历模型的所有参数，返回每个参数的名称和对应的Tensor
    for name, param in model.named_parameters():
        # 如果该参数在加载的state_dict中找到了对应的权重
        if name in state_dict:
            # 将加载的权重复制到模型的对应参数中，并将其转移到指定设备
            param.copy_(state_dict[name].to(device))
        else:
            # 如果没有找到对应的权重，输出警告信息
            print(f"Warning: {name} not found in state_dict.")

# 打印复制完权重后的内存使用情况
print_memory_usage()


# 将输入张量 test_input 创建为一个包含一个样本的张量，形状为 (1, 3)，并将其移动到指定的设备（如 GPU）
test_input = torch.tensor([[1, 2, 3]]).to(device)

# 设置模型为评估模式，意味着在推理过程中不启用梯度计算，例如禁用 dropout 或 batch normalization 的更新
model.eval()

# 在没有计算梯度的上下文环境中执行模型的前向传播，torch.no_grad() 用于提高推理时的计算效率
with torch.no_grad():
    # 将 test_input 输入到模型中并进行推理（即前向传播）
    model(test_input)

# 删除模型对象，释放内存
del model

# 删除输入张量 test_input，释放内存
del test_input

# 删除模型的状态字典 state_dict，通常包含模型的权重等信息
del state_dict

# 删除其他可能定义的参数 param，释放内存
del param

# 调用 cleanup 函数进行进一步的清理操作，例如释放其他资源
cleanup()













import os  # 导入os模块，提供操作系统功能
import psutil  # 导入psutil模块，用于获取系统和进程的内存和CPU等信息
from threading import Thread  # 从threading模块导入Thread类，用于多线程操作


def memory_usage_in_gb(func, *args, **kwargs):
    """
    函数：计算并返回执行给定函数时占用的最大内存（以GB为单位）。
    参数：
    - func: 需要测量内存的函数。
    - *args, **kwargs: func函数的其他参数。
    """
    process = psutil.Process(os.getpid())  # 获取当前进程的psutil对象，用于获取内存使用情况

    # 测量在运行函数之前的内存基准值（单位：GB）
    baseline_mem = process.memory_info().rss / 1024 ** 3  # rss表示进程占用的实际内存，以字节为单位，转换成GB

    # 用于存储内存使用量的列表
    mem_usage = []
    done = False  # 控制内存监控线程是否停止

    def monitor_memory():
        """
        在单独的线程中实时监控内存使用情况。
        """
        while not done:
            # 获取当前内存使用情况（单位：GB）并将其添加到mem_usage列表
            mem_usage.append(process.memory_info().rss / 1024 ** 3)
            time.sleep(0.1)  # 每100ms检查一次内存

    # 创建并启动监控内存使用的线程
    t = Thread(target=monitor_memory)
    t.start()

    # 执行传入的函数
    func(*args, **kwargs)

    # 结束内存监控
    done = True
    t.join()  # 等待监控线程结束

    # 计算函数执行过程中占用的最大内存，减去基准内存值
    peak_mem_usage_gb = max(mem_usage) - baseline_mem
    return peak_mem_usage_gb  # 返回函数执行过程中的最大内存使用量（单位：GB）


# 以下是两个函数，用于加载模型并计算内存使用

def load_sequentially():
    """
    函数：加载模型并按顺序将预训练权重加载到模型的参数中。
    """
    start_memory_tracking()  # 开始内存监控（此函数需提前定义，代码中没有显示）

    model = GPTModel(BASE_CONFIG).to(device)  # 初始化模型，加载基本配置并移动到指定设备（例如GPU）

    # 加载预训练的模型权重
    state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

    print_memory_usage()  # 打印当前的内存使用情况

    # 按顺序将权重加载到模型的每个参数中
    with torch.no_grad():  # 禁用梯度计算（因为只进行推理）
        for name, param in model.named_parameters():  # 遍历模型的每个参数
            if name in state_dict:  # 如果该参数在预训练权重中
                param.copy_(state_dict[name].to(device))  # 将权重复制到模型的参数中，并移至目标设备
            else:
                print(f"Warning: {name} not found in state_dict.")  # 如果该参数在预训练权重中不存在，发出警告

    print_memory_usage()  # 再次打印当前内存使用情况


# 使用memory_usage_in_gb函数测量并输出load_sequentially函数占用的最大内存
peak_memory_used = memory_usage_in_gb(load_sequentially)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 输出加载模型时的最大内存使用量


def baseline():
    """
    函数：使用load_state_dict方法直接加载整个模型的权重。
    """
    start_memory_tracking()  # 开始内存监控

    model = GPTModel(BASE_CONFIG)  # 初始化模型，加载基本配置
    model.to(device)  # 将模型移动到指定设备（例如GPU）

    # 直接加载模型权重
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.to(device)  # 再次将模型移动到目标设备，确保权重已加载
    model.eval()  # 设置模型为评估模式，禁用dropout等训练时的行为

    print_memory_usage()  # 打印当前内存使用情况


# 使用memory_usage_in_gb函数测量并输出baseline函数占用的最大内存
peak_memory_used = memory_usage_in_gb(baseline)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 输出直接加载权重时的最大内存使用量














def best_practices():
    # 使用 "meta" 设备来定义一个上下文，这通常用于示范或其他特殊情况。
    # "meta" 设备并不实际存在，主要用于某些特定目的，比如实验性的模型或设备。
    with torch.device("meta"):
        # 初始化 GPT 模型，假设 BASE_CONFIG 是配置文件，定义了模型的架构、超参数等
        model = GPTModel(BASE_CONFIG)

    # 加载模型的权重参数，"map_location=device" 是将模型加载到指定的设备上
    # "weights_only=True" 表示仅加载权重，不加载其他模型状态信息
    # "mmap=True" 表示启用内存映射文件，这可以节省内存并提高加载速度
    model.load_state_dict(
        torch.load("model.pth", map_location=device, weights_only=True, mmap=True),
        assign=True
    )

    # 打印内存使用情况，通常用于调试，检查加载模型过程中内存的占用情况
    print_memory_usage()

# 调用 memory_usage_in_gb 函数并传入 best_practices 函数作为参数
# 它会计算并返回在执行 best_practices 时所使用的最大内存（以 GB 为单位）
peak_memory_used = memory_usage_in_gb(best_practices)

# 输出最大内存占用的结果，保留一位小数
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")













# 假设 `model` 是你已经训练好的模型
state_dict = model.state_dict()  # 获取模型的所有参数字典

# 创建一个目录用来存储各个参数的文件
os.makedirs("model_parameters", exist_ok=True)  # 如果目录不存在，则创建目录

# 将每个参数张量单独保存到文件中
for name, param in state_dict.items():  # 遍历模型的每个参数
    torch.save(param.cpu(), f"model_parameters/{name}.pt")  # 将每个参数保存为 .pt 文件

del model  # 删除模型对象，释放内存


def load_individual_weights():
    # 启动内存跟踪
    start_memory_tracking()

    # 在“meta”设备上创建模型（这个设备用于管理大模型，通常不会占用实际GPU内存）
    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)  # 初始化模型

    # 将模型转移到指定设备（例如，GPU）并返回一个空的模型
    model = model.to_empty(device=device)

    # 打印当前内存使用情况
    print_memory_usage()
    param_dir = "model_parameters"  # 设置存放模型参数的目录

    # 在不计算梯度的情况下进行参数加载
    with torch.no_grad():
        # 遍历模型的命名参数
        for name, param in model.named_parameters():
            # 为每个参数构造权重文件的路径
            weight_path = os.path.join(param_dir, f"{name}.pt")
            # 检查权重文件是否存在
            if os.path.exists(weight_path):
                # 加载权重数据到 CPU
                param_data = torch.load(weight_path, map_location="cpu", weights_only=True)
                # 将加载的数据拷贝到模型参数中
                param.copy_(param_data)
                # 删除 param_data 以释放内存
                del param_data
            else:
                print(f"Warning: {name} not found in {param_dir}.")  # 警告：未找到参数文件

    # 打印当前内存使用情况
    print_memory_usage()

# 测量执行 load_individual_weights 函数时的最大 CPU 内存使用量
peak_memory_used = memory_usage_in_gb(load_individual_weights)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印最大内存使用情况

# 记录加载权重时的内存峰值
peak_memory_used = memory_usage_in_gb(load_individual_weights)  # 获取加载权重过程中使用的最大内存（假设 `memory_usage_in_gb` 是定义好的函数）
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")  # 打印最大内存使用量