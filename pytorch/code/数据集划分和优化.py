import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


batch_size=200
learning_rate=0.01
epochs=10

train_db = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))#加载数据集

# 图像归一化：
# normalize=transforms.Normalize(mean=[0,485,0.465,0.406],std=[0.229,0.224,0.225])#rgb三通道


#训练集
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)

test_db = datasets.MNIST('../data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))#加载测试集
#测试集
test_loader = torch.utils.data.DataLoader(test_db,batch_size=batch_size, shuffle=True)


print('train:', len(train_db), 'test:', len(test_db))
#将训练集分为训练集和验证集
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print('db1:', len(train_db), 'db2:', len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)#形成新的训练集
val_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size, shuffle=True)#形成验证集




class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
# optimizer=optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.01)#weight_decay参数表示L2正则化中lambda的值
#动量优化
# optimizer=torch.optim.SGD(net.parameters(),learning_rate,momentum=0.2,weight_decay=0.01)
# #参数moment为设置动量法中上一次梯度占的比重

#学习率

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')#在平坦区域学习率衰减
criteon = nn.CrossEntropyLoss().to(device)



for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        # L1 正则化
        # regularization_loss = 0
        # for param in net.parameters():
        #     regularization_loss += torch.sum(torch.abs(param))
        # classify_loss = criteon(logits, target)
        # loss = classify_loss + 0.01 * regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    #验证集
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

#加载验证集performance最好的
#测试集，测试集只验证一次
test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1, 28 * 28)
    data, target = data.to(device), target.cuda()
    logits = net(data)
    test_loss += criteon(logits, target).item()

    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))