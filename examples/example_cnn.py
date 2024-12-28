import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace")
if "/workspace/torch" in sys.path:
    sys.path.remove("/workspace/torch")
    
# 將本地的 conv.py 路徑加入 sys.path
module_path = os.path.abspath("/workspace/torch/nn/modules")
sys.path.insert(0, module_path)

# 匯入修改後的 conv.py
import conv as conv_modules

print(f"Conv2d module path: {conv_modules.__file__}")

# 定義簡單的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 使用本地修改過的 Conv2d 模組
        self.conv1 = conv_modules.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_modules.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
model = SimpleCNN()

# 驗證模型結構
print("Model structure:")
print(model)

# 測試模型是否可執行
dummy_input = torch.randn(1, 1, 28, 28)  # 單張 28x28 的灰度影像
output = model(dummy_input)
print(f"Output shape: {output.shape}")
