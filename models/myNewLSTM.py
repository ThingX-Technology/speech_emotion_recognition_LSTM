import torch
import torch.nn as nn

# 定义卷积网络用于特征提取
class AudioFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(AudioFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(64, output_dim)  # 提取固定维度的嵌入

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=2)  # 全局池化
        x = self.fc(x)
        return x

# 定义完整模型：特征提取 + LSTM 分类器
class AudioEmotionClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, num_classes):
        super(AudioEmotionClassifier, self).__init__()
        self.feature_extractor = AudioFeatureExtractor(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x)
        x = x.unsqueeze(1)  # 添加时间维度 (batch, seq_len=1, embedding_dim)
        # LSTM 分类
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
