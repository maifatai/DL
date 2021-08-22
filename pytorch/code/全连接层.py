import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

#全连接层
#第一种表示
x=torch.randn([10,784])
layer1=nn.Linear(784,200)
layer2=nn.Linear(200,200)
layer3=nn.Linear(200,10)

tmp1=layer1(x)
tmp2=layer2(tmp1)
tmp3=layer3(tmp2)

#第二种表示
def forward(inputs):
    x=layer1(inputs)
    x=F.relu(x,inplace=True)#参数inplace表示输入输出维度一致
    x=layer2(inputs)
    x=F.relu(x,inplace=True)
    x=layer2(inputs)
    x=F.relu(x,inplace=True)

#第三种表示
net1=nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(inplace=True),
    nn.Linear(200, 200),
    nn.ReLU(inplace=True),
    nn.Linear(200, 10),
    nn.ReLU(inplace=True),
)


#第四种表示，自定义网络
class MLP(nn.Module):
# '''
# 1.从nn.Module继承
# 2.初始化 super(net, self).__init__()
# 3.def forward()
# '''

#relu的两种调用API
# 1.类方法 nn.ReLU
# 2.函数 F.relu()
    def __init__(self):
        super(MLP, self).__init__()
        self.l1=nn.Linear(784, 200)
        self.act1=nn.ReLU(inplace=True)
        self.l2=nn.Linear(200,200)
        self.act2=nn.ReLU(inplace=True)
        self.l3=nn.Linear(200,10)
        self.act2=nn.ReLU(inplace=True)
    def forward(self, x):
        x=self.l1(x)
        x=self.act1(x)
        x=self.l2(x)
        x=self.act2(x)
        x=self.l3(x)
        out=self.act3(x)
        return out
net = MLP()
#net.parameters()表示网络的可训练的参数
optimizer = optim.SGD(net.parameters(), lr=learning_rate),
criteon = nn.CrossEntropyLoss()#损失函数
#训练过程
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net(data)
        test_loss += criteon(logits, target).item()#可以不用计算

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()#eq转换为0,1，然后sum求和

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
