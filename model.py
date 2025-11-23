import torch
import torch.nn as nn


class ClaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.liner_model = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()
        )
        
    def forward(self,x):
        x = self.flatten(x)
        logits = self.liner_model(x)
        return logits


# import torch

# # GPUが利用可能かチェック
# is_available = torch.cuda.is_available()
# print(f"CUDA Available: {is_available}")

# if is_available:
#     # 認識されているGPU名を表示
#     print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
#     # RTX 3060は 'Ampere' アーキテクチャなので、対応した計算能力も表示してみる
#     print(f"Capability: {torch.cuda.get_device_capability(0)}")
# else:
#     print("GPUが認識されていません。CPU版のPyTorchが入っている可能性があります。")