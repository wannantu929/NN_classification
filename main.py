import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import model
import train

def main():
    
    # 訓練データのダウンロードと読み込み 
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,       
        download=True,
        transform=ToTensor(), 
    )

    # テストデータのダウンロードと読み込み 
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,      
        download=True,
        transform=ToTensor(),
    )

    # データローダーの作成（バッチサイズごとにデータをまとめて供給する）
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print(f"Training Data Size: {len(training_data)}")
    print(f"Test Data Size: {len(test_data)}")

    # GPUが使えるか確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    cla_model = model.ClaModel().to(device)
    
    cross_loss = nn.CrossEntropyLoss()#他クラス分類
    optimizer  = torch.optim.SGD(cla_model.parameters(), lr=1e-3) # 最適化アルゴリズム
    
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train.train(train_dataloader, cla_model,cross_loss,optimizer,device)
        train.test(test_dataloader, cla_model,cross_loss,device)

if __name__ == "__main__":
    main()