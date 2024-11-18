# import os
import numpy as np
import librosa
import torch
import joblib
# from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn

from models.myLSTM import EmotionLSTM

# 加载标准化器和标签编码器
scaler = joblib.load('checkpoint/scaler.joblib')
le = joblib.load('checkpoint/label_encoder.joblib')


input_size = 40
hidden_size = 128
num_layers = 3
num_classes = len(le.classes_)  # 情感类别数量
print(num_classes)
dropout = 0.5

# 实例化模型
model = EmotionLSTM(input_size, hidden_size, num_layers, num_classes, dropout)

# 加载模型的 state_dict
model.load_state_dict(torch.load('checkpoint/emotion_best_model.pth', map_location=torch.device('cpu')))
model.eval()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 特征提取函数
def extract_features(file_path, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 情感预测函数
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return None

    # 应用与训练时相同的标准化
    features = scaler.transform([features])  # features 形状为 (1, 40)

    # 转换为张量
    features = torch.tensor(features, dtype=torch.float32).to(device)  # 形状为 (1, 40)
    features = features.unsqueeze(1)  # 添加序列维度，形状变为 (1, 1, 40)

    # 检查形状
    print(f"Features shape: {features.shape}")  # 应该输出 torch.Size([1, 1, 40])

    # 执行推理
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = le.inverse_transform(predicted.cpu().numpy())
        return predicted_label[0]


if __name__ == '__main__':

    audio_file = 'audios/angry.wav'
    emotion = predict_emotion(audio_file)
    if emotion:
        print(f'Predicted Emotion: {emotion}')
    else:
        print('Could not predict emotion due to an error in feature extraction.')
