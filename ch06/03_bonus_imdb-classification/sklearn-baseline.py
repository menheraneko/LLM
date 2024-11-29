import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# 加载数据集
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

train_df.head() # 展示数据头

# 创建一个CountVectorizer实例，用于将文本转换为词频特征
vectorizer = CountVectorizer()

# 使用训练数据的文本生成词频特征矩阵X_train，并拟合词汇表
X_train = vectorizer.fit_transform(train_df["text"])  # 从train_df中提取文本列并生成特征矩阵

# 使用验证数据的文本转换为词频特征矩阵X_val，基于之前拟合的词汇表
X_val = vectorizer.transform(val_df["text"])  # 从val_df中提取文本列并生成特征矩阵

# 使用测试数据的文本转换为词频特征矩阵X_test，基于之前拟合的词汇表
X_test = vectorizer.transform(test_df["text"])  # 从test_df中提取文本列并生成特征矩阵

# 提取标签数据，训练、验证和测试集的标签分别存储在y_train、y_val和y_test中
y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]



def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Making predictions
    y_pred_train = model.predict(X_train) # 训练集预测
    y_pred_val = model.predict(X_val) # 验证集
    y_pred_test = model.predict(X_test) # 测试集

    # Calculating accuracy and balanced accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train) # 计算准确分数
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train) # 平衡准确分数

    accuracy_val = accuracy_score(y_val, y_pred_val) # 验证集评估
    balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    accuracy_test = accuracy_score(y_test, y_pred_test) # 测试集评估
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # 输出结果
    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")





# 创建一个dummy分类器，使用“最频繁”的策略（即总是预测最常见的类别）
dummy_clf = DummyClassifier(strategy="most_frequent")
# 在训练数据上拟合dummy分类器
dummy_clf.fit(X_train, y_train)

# 使用eval函数评估dummy分类器的性能
eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

# 创建逻辑回归模型，最大迭代次数设置为1000
model = LogisticRegression(max_iter=1000)
# 在训练数据上拟合逻辑回归模型
model.fit(X_train, y_train)

# 使用eval函数评估逻辑回归模型的性能，传递训练、验证和测试集
eval(model, X_train, y_train, X_val, y_val, X_test, y_test)