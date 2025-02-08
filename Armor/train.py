import os
import torch
from torch import nn
from torchvision import transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
import output

def train(net, train_iter, test_iter, num_epochs, lr, save=False, save_path="./digit.pth"):
    """
    训练并保存模型，最后展示训练效果折线图
    
    参数:
    net: 神经网络模型
    train_iter: 训练数据迭代器
    test_iter: 测试数据迭代器
    num_epochs: 训练轮数
    lr: 学习率
    save: 是否保存模型, 默认为False
    save_path: 模型保存路径，默认为"./digit.pth"

    """
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    i = 1
    
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        
        net.train()
        metric = d2l.Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], (y_hat.argmax(dim=1) == y).sum().item(), y.numel())
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        
        net.eval()
        metric = d2l.Accumulator(2)
        for X, y in test_iter:
            metric.add((net(X).argmax(dim=1) == y).sum().item(), y.numel())
        test_acc =  metric[0] / metric[1]
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        print(f'{i}, train loss {train_l:.3f}, train acc {train_acc:.3f}, ' f'test acc {test_acc:.3f}', f'{timer.stop():.1f} sec')
        i=i+1
        
    if(save):
        torch.save(net.state_dict(), save_path)
    plt.show()


def show(imgs, labels, preds, n=8):
    """
    展示测试效果

    参数:
    imgs: 输入图像列表
    labels: 真实标签列表
    preds: 预测标签列表
    n: 展示的图像数量, 默认为8

    """
    fig, axes = plt.subplots(1, n, figsize=(12, 12 // n))
    for i in range(n):
        axes[i].imshow(imgs[i].reshape(28, 28).numpy())
        axes[i].set_title(f"Label: {labels[i]}\nPred: {preds[i]}")
        axes[i].axis('off')
    plt.show()
    
def load_model(net, load_path="model.pth"):
    """加载模型参数"""
    if os.path.exists(load_path):
        net.load_state_dict(torch.load(load_path))
        net.eval()
        print(f"Model loaded from {load_path}")
    else:
        print(f"Model file not found at {load_path}")

def test(test_iter, net):
    """测试模型效果"""
    X, y = next(iter(test_iter))
    X, y = X[:8], y[:8]
    Logits = net(X)
    prob = torch.nn.functional.softmax(Logits, dim=1)
    preds = prob.argmax(dim=1).numpy()
    show(X, y.numpy(), preds)

def dataload(batch_size, train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

    valid_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset,train_loader,valid_loader

if __name__ == "__main__":
    batch_size = 32
    lr = 0.1
    epochs = 25
    net = nn.Sequential(nn.Flatten(), nn.Linear(400, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 5))

    train_dir = './dataset/train'
    valid_dir = './dataset/valid'
    test_dir = './dataset/test'

    train_dataset, train_loader, valid_loader = dataload(batch_size, train_dir, valid_dir, test_dir)

    classes = train_dataset.classes
    print("类别标签:", classes)
    
    train(net, train_loader, valid_loader, epochs, lr, save=True)
    output.export(net, r'./digit.pth', input_shape=(1, 1, 20, 20), onnxfile=r'./digit.onnx', local='cpu')
