import torch





# a9.1
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 创建两个张量 tensor_1 和 tensor_2
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

# 打印 tensor_1 和 tensor_2 的加法结果
print(tensor_1 + tensor_2)

# 将 tensor_1 和 tensor_2 转换到 CUDA 设备 (GPU)
tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")

# 打印在 GPU 上的加法结果
print(tensor_1 + tensor_2)

# 将 tensor_1 转回到 CPU，并打印 tensor_1 和 tensor_2 的加法结果
tensor_1 = tensor_1.to("cpu")
print(tensor_1 + tensor_2)



# a9.2
# 创建训练数据集
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

# 创建标签
y_train = torch.tensor([0, 0, 0, 1, 1])

# 测试集
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

# 创建标签
y_test = torch.tensor([0, 1])


#  ToyDataset
class ToyDataset(Dataset):
    def __init__(self, X, y):
        # 初始化特征和标签
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        # 返回指定索引的特征和标签
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

    # 创建训练和测试数据集的实例


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
torch.manual_seed(123)  # 设置随机种子以确保可重复性

# 训练loader
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=1,
    drop_last=True   # drop最后元素
)

# 测试loader
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,  # 每个批次的大小
    shuffle=False,  # 不打乱数据
    num_workers=1  # 使用的子进程数量
)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # 网络结构
        self.layers = torch.nn.Sequential(
            # 第一隐藏层
            torch.nn.Linear(num_inputs, 30), # 线性层
            torch.nn.ReLU(), # 激活函数
            # 第二隐藏层
            torch.nn.Linear(30, 20), # 线性层
            torch.nn.ReLU(), # 激活函数
            # 输出层
            torch.nn.Linear(20, num_outputs), # 线性层
        )

    def forward(self, x):
        # 前向传播
        logits = self.layers(x)
        return logits



torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)  # 实例化模型

# 检查 CUDA 是否可用，以选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 将模型移动到设备

# 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3  # 训练的轮数

# 进行训练
for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)  # 移动特征和标签到设备
        logits = model(features)  # 前向传播得到预测结果
        loss = F.cross_entropy(logits, labels)  # 计算损失

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印每个批次的训练信息
        print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()  # 设置模型为评估模式


def compute_accuracy(model, dataloader, device):

    model = model.eval()  # 评估模式
    correct = 0.0  # 正确预测的计数
    total_examples = 0  # 总样本数

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)  # 移动特征和标签到设备

        with torch.no_grad():  # 禁用梯度
            logits = model(features)  # 预测结果

        predictions = torch.argmax(logits, dim=1)  # 获取类别
        compare = labels == predictions  # 比较预测与真实标签
        correct += torch.sum(compare)  # 正确预测的数量累加
        total_examples += len(compare)  # 总样本数累加

    return (correct / total_examples).item()  # 计算准确率并返回


# 计算训练集和测试集的准确率
compute_accuracy(model, train_loader, device=device)
compute_accuracy(model, test_loader, device=device)




