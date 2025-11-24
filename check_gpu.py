import torch

# GPUが利用可能かチェック
is_available = torch.cuda.is_available()
print(f"CUDA Available: {is_available}")

if is_available:
    # 認識されているGPU名を表示
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    

    print(f"Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("GPUが認識されていません。CPU版のPyTorchが入っている可能性があります。")