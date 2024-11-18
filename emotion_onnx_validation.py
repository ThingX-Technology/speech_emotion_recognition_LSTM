import torch
import onnx
import onnxruntime as ort
import numpy as np
import joblib
import json
from models.myLSTM import EmotionLSTM

# 加载 scaler 和 label encoder
with open('scaler_params.json', 'r') as f:
    scaler_params = json.load(f)
with open('label_classes.json', 'r') as f:
    label_classes = json.load(f)

# 加载原始PyTorch模型
input_size = 40
hidden_size = 128
num_layers = 3
num_classes = len(label_classes)  # 情感类别数量
dropout = 0.5

# 实例化模型
model = EmotionLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
model.load_state_dict(torch.load('checkpoint/emotion_best_model.pth', map_location=torch.device('cpu')))
model.eval()

# 准备测试输入
dummy_input = torch.randn(1, 1, input_size)  # (batch_size, sequence_length, input_size)

# 使用原始模型进行预测
with torch.no_grad():
    pytorch_output = model(dummy_input).numpy()
    print("PyTorch 模型输出:", pytorch_output)

# 检查ONNX模型
onnx_model = onnx.load("thingx_emotion_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 模型已通过验证！")

# 使用ONNX Runtime进行推理
ort_session = ort.InferenceSession("thingx_emotion_model.onnx")

# 准备输入
ort_inputs = {"input": dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
onnx_output = ort_outs[0]
print("ONNX 模型输出:", onnx_output)

# 对比原始模型输出和ONNX模型输出
if np.allclose(pytorch_output, onnx_output, atol=1e-5):
    print("ONNX模型输出与PyTorch模型输出一致！")
else:
    print("ONNX模型输出与PyTorch模型输出不一致。")
