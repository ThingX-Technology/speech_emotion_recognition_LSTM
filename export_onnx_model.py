import torch
from models.myLSTM import EmotionLSTM
import joblib
import json

# 加载标准化器和标签编码器
scaler = joblib.load('checkpoint/scaler.joblib')
le = joblib.load('checkpoint/label_encoder.joblib')

# 提取均值和标准差
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}

# 保存为JSON文件
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)

print("Scaler参数已保存到 scaler_params.json")


# 提取类标签
label_classes = le.classes_.tolist()

# 保存为JSON文件
with open('label_classes.json', 'w') as f:
    json.dump(label_classes, f)

print("标签类已保存到 label_classes.json")



input_size = 40
hidden_size = 128
num_layers = 3
num_classes = len(le.classes_)  # 情感类别数量
dropout = 0.5


model = EmotionLSTM(input_size, hidden_size, num_layers, num_classes, dropout)

# 加载模型的 state_dict
model.load_state_dict(torch.load('checkpoint/emotion_best_model.pth', map_location=torch.device('cpu')))
model.eval()


dummy_input = torch.randn(1, 1, input_size)  # (batch_size, sequence_length, input_size)

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "thingx_emotion_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"}, # "input": {0: "batch_size", 1: "sequence_length"} 表示输入张量的第0维（通常是批量大小）和第1维（例如序列长度）是动态的，可以在不同推理过程中变化。
        "output": {0: "batch_size"}  # "output": {0: "batch_size"} 表示输出张量的第0维（批量大小）是动态的。
    },
    opset_version=11  # 指定ONNX算子集的版本
)

print("模型已成功导出为 thingx_emotion_model.onnx")
