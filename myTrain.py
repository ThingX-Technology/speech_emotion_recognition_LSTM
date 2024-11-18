import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 用于保存和加载 scaler 和 label encoder
from models.myLSTM import EmotionLSTM  # 请确保您有这个模型文件
from myNewDataset import load_data, EmotionDataset  # 使用修改后的 dataset.py

# 数据加载和预处理
data_path = 'datasets'  # 请替换为您的实际数据集路径
X, y = load_data(data_path)

# 标准化特征并保存 scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'checkpoint/scaler.joblib')  # 保存标准化器

# 编码标签并保存 label encoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'checkpoint/label_encoder.joblib')  # 保存标签编码器

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 创建 Dataset
train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)

# DataLoader 定义
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型定义
input_size = 40  # 输入特征维度
hidden_size = 128
num_layers = 3
num_classes = len(le.classes_)  # 情感类别数量
dropout = 0.5
model = EmotionLSTM(input_size, hidden_size, num_layers, num_classes, dropout)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5000  # 您可以根据需要调整训练轮数
best_accuracy = 0.0

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

    for inputs, labels in loop:
        inputs = inputs.to(device).unsqueeze(1)  # 添加序列维度
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条显示当前的损失和准确率
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct / total
    print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')

    # 测试评估
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).unsqueeze(1)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    print(f'Test Accuracy: {test_acc:.2f}%')

    # 保存最优模型
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        # 保存最优模型 模型命名中添加准确率
        torch.save(model.state_dict(), f'emotion_best_model_{test_acc:.2f}.pth')
        # torch.save(model.state_dict(), 'emotion_best_model.pth')
        print('Best model saved.')

# 加载最优模型并进行最终评估
model.load_state_dict(torch.load('checkpoint/emotion_best_model.pth'))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device).unsqueeze(1)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 分类报告
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
