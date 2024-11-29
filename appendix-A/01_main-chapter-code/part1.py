import torch
from torch.nn import Sequential, Linear, ReLU

# a1
print(torch.__version__)

print(torch.cuda.is_available())





# a2.1
import numpy as np


tensor0d = torch.tensor(1) # 创建0维张量，为一个点
tensor1d = torch.tensor([1, 2, 3]) # 创建1维张量
tensor2d = torch.tensor([[1, 2],
                         [3, 4]]) # 创建二维张量

tensor3d_1 = torch.tensor([[[1, 2], [3, 4]],
                           [[5, 6], [7, 8]]]) # 创建3维张量

# create a 3D tensor from NumPy array
ary3d = np.array([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]]) # 通过numpy数组创建三维张量

tensor3d_2 = torch.tensor(ary3d)  # 复制numpy数组张量
tensor3d_3 = torch.from_numpy(ary3d)  # 和numpy张量共享记忆

ary3d[0, 0, 0] = 999
print(tensor3d_2) # 复制，张量不变

print(tensor3d_3) # 共享记忆，因原数组改变而改变



# a2.2
tensor1d = torch.tensor([1, 2, 3]) # int64类型张量
print(tensor1d.dtype) # 输出

floatvec = torch.tensor([1.0, 2.0, 3.0]) # float32类型
print(floatvec.dtype) # 输出

floatvec = tensor1d.to(torch.float32) # 数据类型转换，int64到float32
print(floatvec.dtype) # 输出





# a2.3
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]]) # 二维张量

tensor2d.reshape(3, 2) # 重整形状，按照列展开，从小到大排列形成新维度
tensor2d.view(3, 2) # 重整形状，要求内存连续

tensor2d.matmul(tensor2d.T) # 自身乘自身转置
tensor2d @ tensor2d.T # 矩阵乘法






# a3
import torch.nn.functional as F

y = torch.tensor([1.0])  # 实际 label
x1 = torch.tensor([1.1]) # 输入特征
w1 = torch.tensor([2.2]) # 权重参数
b = torch.tensor([0.0])  # bias

z = x1 * w1 + b          # 神经网络输出计算，input*weight+bias
a = torch.sigmoid(z)     # 激活函数

loss = F.binary_cross_entropy(a, y) # 二元交叉熵loss
print(loss) # 输出loss





# a4
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0]) # 目标
x1 = torch.tensor([1.1]) # input
w1 = torch.tensor([2.2], requires_grad=True) # 权重参数
b = torch.tensor([0.0], requires_grad=True) # bias

z = x1 * w1 + b # output计算
a = torch.sigmoid(z) # 激活函数

loss = F.binary_cross_entropy(a, y) # 二元交叉熵

grad_L_w1 = grad(loss, w1, retain_graph=True) # 计算loss到权重的梯度
grad_L_b = grad(loss, b, retain_graph=True) # 计算到bias的梯度

print(grad_L_w1) # 输出
print(grad_L_b)

loss.backward() # 反向传播梯度

print(w1.grad) # 输出梯度
print(b.grad) # bias









# a5
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): # 初始化
        super().__init__()

        self.layers = torch.nn.Sequential( # 建立layer

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30), # 线形层
            torch.nn.ReLU(), # 激活函数

            # 2nd hidden layer
            torch.nn.Linear(30, 20), # 线形层
            torch.nn.ReLU(), # 激活函数

            # output layer
            torch.nn.Linear(20, num_outputs), # 线形层
        )

    def forward(self, x): # 前向传播
        logits = self.layers(x)
        return logits



model = NeuralNetwork(50, 3) # 创建模型对象

print(model) # 输出模型信息



num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 如果允许梯度传播，记录。则输出参数个数
print("Total number of trainable model parameters:", num_params) # 输出

print(model.layers[0].weight) # 第0层权重

torch.manual_seed(123) # 种子

model = NeuralNetwork(50, 3) # 创建模型
print(model.layers[0].weight) # 输出该层权重
torch.manual_seed(123)

X = torch.rand((1, 50)) # 测试示例
out = model(X) # 获取输出
print(out) # 输出

with torch.no_grad():
    out = model(X) # 禁用梯度，获取输出
print(out)

with torch.no_grad():
    out = torch.softmax(model(X), dim=1) # 归一化
print(out)






# a6
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
]) # 训练示例

y_train = torch.tensor([0, 0, 0, 1, 1]) # 目标输出

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
]) # 测试示例

y_test = torch.tensor([0, 1]) # 测试输出

from torch.utils.data import Dataset


class ToyDataset(Dataset): # dataset
    def __init__(self, X, y):
        self.features = X # 特征
        self.labels = y # 标签

    def __getitem__(self, index):
        one_x = self.features[index] # 获取特征
        one_y = self.labels[index]  # 获取标签
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train) # 获取训练数据
test_ds = ToyDataset(X_test, y_test) # 测试

from torch.utils.data import DataLoader

torch.manual_seed(123) # 种子

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
) # loader

test_ds = ToyDataset(X_test, y_test) # 测试数据

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
) # loader

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y) # 输出batch信息






# a7
import torch.nn.functional as F

torch.manual_seed(123) # 种子
model = NeuralNetwork(num_inputs=2, num_outputs=2) # 获取模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.5) # 优化器，学习率0.5

num_epochs = 3 # 训练轮数

for epoch in range(num_epochs):

    model.train() # 训练模式
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features) # 获取输出

        loss = F.cross_entropy(logits, labels)  # 计算交叉熵

        optimizer.zero_grad() # 梯度清0
        loss.backward() # 反向传播
        optimizer.step() # 迭代

        ### LOGGING
        print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}") # 输出训练信息

    model.eval() # 切换评估
    # Optional model evaluation


model.eval() # 评估

with torch.no_grad():
    outputs = model(X_train) # 禁用梯度评估

print(outputs) # 输出


torch.set_printoptions(sci_mode=False) # 禁用科学计数法
probas = torch.softmax(outputs, dim=1) # 归一化
print(probas) # 输出

predictions = torch.argmax(probas, dim=1) # 取预测最大概率
print(predictions) # 输出

predictions = torch.argmax(outputs, dim=1) # 取输出最大分布
print(predictions) # 输出


torch.sum(predictions == y_train) # 预测成功的总数


def compute_accuracy(model, dataloader): # 计算准确率
    model = model.eval() # 评估
    correct = 0.0 # 正确
    total_examples = 0 # 总数

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features) # 获取分布

        predictions = torch.argmax(logits, dim=1) # 贪心选择概率最大的
        compare = labels == predictions # 比较batch中元素是否正确，记录
        correct += torch.sum(compare) # 统计正确总数
        total_examples += len(compare) # 总长度++

    return (correct / total_examples).item() # 返回准确率

compute_accuracy(model, train_loader) # 调用评估



# a8

torch.save(model.state_dict(), "model.pth") # 保存模型

model = NeuralNetwork(2, 2) # 建立模型，基本形状要和原模型相同
model.load_state_dict(torch.load("model.pth", weights_only=True)) # 加载模型






