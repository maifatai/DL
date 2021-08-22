import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from visdom import Visdom
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(400,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,10),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        logits= self.net(x)
        return logits
#测试
# net=Lenet()
# input=torch.randn(2,3,32,32)
# out=net(input)
# print(out.shape)
#加载数据集
cifar_train=datasets.CIFAR10('cifar10',#在当前目录下
                           train=True,
                           transform=transforms.Compose(
                               [transforms.Resize((32,32)),
                                transforms.ToTensor()]
                           ),download=True)#一次只能加载一个
cifar_load=DataLoader(dataset=cifar_train,batch_size=32,shuffle=True)
cifar_test=datasets.CIFAR10('cifar10',#在当前目录下
                           train=False,
                           transform=transforms.Compose(
                               [transforms.Resize((32,32)),
                                transforms.ToTensor()]
                           ),download=True)#一次只能加载一个
cifar_load_test=DataLoader(dataset=cifar_test,batch_size=32,shuffle=True)
model=Lenet()
criteon=nn.CrossEntropyLoss()
viz = Visdom(env='demo')
optimize=torch.optim.Adam(model.parameters(),lr=1e-3)
for epoch in range(1000):
    for batch_idx,(x,y) in  enumerate(cifar_load):
        '''
        1.网络前向
        2.计算损失函数
        3.梯度清零
        4.loss.backward()
        5.梯度计算
        '''
        logits=model(x)
        loss=criteon(logits,y)
        optimize.zero_grad()#optimize梯度清零
        loss.backward()#梯度计算
        optimize.step()#梯度更新，写进optimize

    print('epoch',epoch,'loss:',loss.item())
    viz.line(loss.item(),epoch,win='loss',opts=())


