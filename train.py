import torch

def train(dataloader,model,cross_loss,optimizer,device):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = cross_loss(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader,model,cross_loss,device):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += cross_loss(pred,y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= num_batch
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")