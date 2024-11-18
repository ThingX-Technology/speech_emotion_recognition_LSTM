import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# 提取音频特征
def extract_features(file_path, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 加载数据
def load_data(data_path):
    features = []
    labels = []
    for emotion_label in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion_label)
        if not os.path.isdir(emotion_path):
            continue
        for file in os.listdir(emotion_path):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_path, file)
                mfcc = extract_features(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(emotion_label)  # 使用文件夹名称作为标签
    return np.array(features), np.array(labels)

# 定义 Dataset
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = labels  # 暂时保留为字符串，稍后在训练代码中进行编码

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
