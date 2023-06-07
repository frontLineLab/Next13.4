import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from nn import NeuralNetwork

is_save = False
batch_size = 100
epoch_num = 10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 学習データをダウンロードする
training_data = datasets.FashionMNIST(
    root="./data",        # 保存先のパスを指定する
    train=True,           # 学習データを取得する
    download=True,        # ローカルにない場合はダウンロードする
    transform=ToTensor(), # Tensor に変換（画素値：uint8 → float32）
)

# テストデータをダウンロードする
test_data = datasets.FashionMNIST(
    root="data",          # 保存先のパスを指定する
    train=False,          # テストデータを取得する
    download=True,        # ローカルにない場合はダウンロードする
    transform=ToTensor(), # Tensor に変換（画素値：uint8 → float32）
)

# 学習データ：60000枚
train_dataloader = DataLoader(training_data, batch_size=batch_size)
# 学習データ：10000枚
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# モデル
model = NeuralNetwork().to(device)
# 損失関数
loss = nn.CrossEntropyLoss()
# オプティマイザー
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_train, optimizer):

    model.train()
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # 順伝搬
        pred = model.forward(x)
        # 損失を算出する
        loss = loss_train(pred, y)
        # 誤差逆伝播
        loss.backward()
        # パラメータを更新する
        optimizer.step()
        # パラメータの勾配を０にする
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_test):

    size = len(dataloader.dataset)
    batches_num = len(dataloader)
    model.eval()
    loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # 順伝搬
            pred = model(x)
            # 損失を算出する
            loss += loss_test(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= batches_num
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

def saveModel():
    torch.save(model.state_dict(), "model.pth")
    print("Model Saved")

if __name__ == '__main__':

    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss, optimizer)
        test(test_dataloader, model, loss)

    if(is_save):
        saveModel()