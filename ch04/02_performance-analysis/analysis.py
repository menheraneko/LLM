import torch
from thop import profile

from previous_chapters import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
} # 参数配置

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
} # 参数配置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device) # 测试示例

for size in model_configs:
    BASE_CONFIG.update(model_configs[size]) # 更新参数

    model = GPTModel(BASE_CONFIG).bfloat16() # 创建模型
    model.to(device) # 转移

    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False) # macs
    flops = 2*macs
    print(f"{size:18}: {flops:.1e} FLOPS") # 计算结果

    del model
    torch.cuda.empty_cache()


for size in model_configs:
    print(f"\nProcessing {size}")
    config = BASE_CONFIG.copy() # copy
    config.update(model_configs[size]) # 更新

    min_batch_size = 1 # 最小大小
    max_batch_size = None # 无最大
    max_possible_batch_size = 4096

    # 二分查找批量最大
    while min_batch_size <= max_possible_batch_size:
        batch_size = (min_batch_size + max_possible_batch_size) // 2
        try:
            input_tensor = torch.randint(
                0, config["vocab_size"],
                (batch_size, config["context_length"]),
                device=device
            ) # 创建输入张量

            model = GPTModel(config).bfloat16().to(device) # 模型

            # MACS = multiply-accumulate operations
            # MACS are typically counted as two FLOPS (one multiply and one accumulate)
            macs, params = profile(model, inputs=(input_tensor,), verbose=False) # macs
            flops = 2 * macs
            print(f"  Batch size {batch_size}: {flops:.1e} FLOPS") # 输出

            # If successful, try a larger batch size
            min_batch_size = batch_size + 1 # 成功就+1
            max_batch_size = batch_size

            # Clean up
            del model, input_tensor # 清楚数据
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Try smaller batch size
                max_possible_batch_size = batch_size - 1

                # Clean up
                try:
                    del model, input_tensor
                    torch.cuda.empty_cache()
                except NameError:
                    pass
            else:
                raise e


flops_per_second = { # 不同显卡下的设置
    # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
    "H100": {
        torch.float32: 51.22e12,  # 51.22 TFLOPs for FP32 on NVIDIA H100
        torch.float16: 204.9e12,  # 204.9 TFLOPs for FP16 on NVIDIA H100
        torch.bfloat16: 204.9e12
    },
    # https://www.techpowerup.com/gpu-specs/l4.c4091
    "L4": {
        torch.float32: 30.29e12,  # 30.29 TFLOPs for FP32 on NVIDIA L4
        torch.float16: 30.29e12,  # 30.29 TFLOPs for FP16 on NVIDIA L4
        torch.bfloat16: 30.29e12
    },
    # https://www.techpowerup.com/gpu-specs/tesla-t4.c3316
    "T4": {
        torch.float32: 8.1e12,  # 8.1 TFLOPs for FP32 on NVIDIA T4
        torch.float16: 65.13e12,  # 65.13 TFLOPs for FP16 on NVIDIA T4
        torch.bfloat16: 65.13e12
    },
    # https://www.techpowerup.com/gpu-specs/a10g.c3798
    "A10G": {
        torch.float32: 31.52e12,  # 31.52 TFLOPs for FP32 on NVIDIA A10G
        torch.float16: 31.52e12,  # 31.52 TFLOPs for FP16 on NVIDIA A10G
        torch.bfloat16: 31.52e12
    },
    # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
    "A100": {
        torch.float32: 19.49e12,  # 19.49 TFLOPs for FP32 on NVIDIA A100
        torch.float16: 77.97e12,  # 77.97 TFLOPs for FP16 on NVIDIA A100
        torch.bfloat16: 77.97e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621
    "RTX_3080": {
        torch.float32: 29.77e12,  # 29.77 TFLOPs for FP32 on NVIDIA RTX 3080
        torch.float16: 29.77e12,  # 29.77 TFLOPs for FP16 on NVIDIA RTX 3080
        torch.bfloat16: 29.77e12
    },
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
    "RTX_3090": {
        torch.float32: 35.58e12,  # 35.58 TFLOPs for FP32 on NVIDIA RTX 3090
        torch.float16: 35.58e12,  # 35.58 TFLOPs for FP16 on NVIDIA RTX 3090
        torch.bfloat16: 35.58e12
    }
}




import time

def get_gpu_model(flops_per_second_dict): # 获取gpu上的模型
    device_name = torch.cuda.get_device_name(0)
    for model in flops_per_second_dict.keys():
        if model in device_name:
            return model
    return "Unknown"  # Default if no matching model is found


gpu_model = get_gpu_model(flops_per_second) # 获取模型
print("GPU Model:", gpu_model)

if gpu_model != "Unknown":  # 检查GPU模型是否已知
    for size in model_configs:
        print(f"\nProcessing {size}")  # 打印当前处理的模型大小
        config = BASE_CONFIG.copy()  # 复制
        config.update(model_configs[size])  # 更新

        min_batch_size = 1  # 设置最小批量大小为1
        max_batch_size = None  # 初始化最大批量大小
        max_possible_batch_size = 4096  # 设置最大可能的批量大小为4096

        # 使用二分法查找合适的批量大小
        while min_batch_size <= max_possible_batch_size:
            batch_size = (min_batch_size + max_possible_batch_size) // 2  # 计算当前批量大小
            try:
                # 创建输入张量，随机生成词汇表大小范围内的整数
                input_tensor = torch.randint(
                    0, config["vocab_size"],
                    (batch_size, config["context_length"]),
                    device=device
                )

                # 创建模型并转换为bfloat16类型，并将模型移动到指定设备
                model = GPTModel(config).bfloat16().to(device)
                model.train()  # 将模型设置为训练模式

                # 开始计时
                torch.cuda.synchronize()  # 确保CUDA操作完成
                start_time = time.time()  # 记录开始时间

                # 前向和反向传播
                output = model(input_tensor)  # 模型前向传播
                loss = output.sum()  # 计算一个虚拟损失
                loss.backward()  # 反向传播计算梯度

                # 结束计时
                torch.cuda.synchronize()  # 确保CUDA操作完成
                end_time = time.time()  # 记录结束时间

                total_time_seconds = end_time - start_time  # 计算总耗时

                # 计算前向传播的FLOPs
                macs, params = profile(model, inputs=(input_tensor,), verbose=False)  # 计算MACs和参数数量
                flops_forward = 2 * macs  # 假设一个MAC等于两个FLOPs

                # 估算反向传播的FLOPs（通常是前向FLOPs的两倍）
                flops_backward = 2 * flops_forward

                # 总FLOPs为前向和反向传播的FLOPs之和
                total_flops = flops_forward + flops_backward  # 或者 total_flops = flops_forward * 3

                data_type = next(model.parameters()).dtype  # 获取模型参数的数据类型
                max_flops_per_second = flops_per_second[gpu_model].get(data_type, 0)  # 获取该GPU模型的最大FLOPs

                # 计算每秒处理的tokens数量
                tokens_processed = batch_size * config["context_length"]  # 计算处理的tokens总数
                tokens_per_second = tokens_processed / total_time_seconds  # 计算每秒处理的tokens数量

                # 计算每个token的FLOPs
                flops_per_token = total_flops / tokens_processed

                # 计算理论最大每秒tokens数量
                if flops_per_token > 0:
                    theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token  # 计算理论最大tokens每秒
                else:
                    theoretical_max_tokens_per_second = 0  # 避免除以零的情况

                # 计算MFU（模型利用率）
                if theoretical_max_tokens_per_second > 0:
                    mfu = tokens_per_second / theoretical_max_tokens_per_second  # 计算MFU
                else:
                    mfu = 0  # 避免除以零的情况

                # 打印当前批量大小下的每秒tokens数量和MFU
                print(f"  Batch size {batch_size}: Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.4f}")

                # 如果成功，尝试更大的批量大小
                min_batch_size = batch_size + 1  # 增加最小批量大小
                max_batch_size = batch_size  # 更新最大批量大小

                # 清理内存
                del model, input_tensor, output, loss  # 删除不再使用的变量
                torch.cuda.empty_cache()  # 清空CUDA缓存

            except RuntimeError as e:
                if "out of memory" in str(e).lower():  # 如果出现内存不足的错误
                    # 尝试更小的批量大小
                    max_possible_batch_size = batch_size - 1  # 减小最大可能的批量大小

                    # 清理内存
                    try:
                        del model, input_tensor  # 删除不再使用的变量
                        torch.cuda.empty_cache()  # 清空CUDA缓存
                    except NameError:
                        pass  # 如果变量不存在，则跳过
                else:
                    raise e  # 重新抛出其他异常

else:
    print("Unknown GPU model. Please update the flops_per_second dictionary with your GPU information.")  # 如果GPU模型未知，打印提示信息