import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from visdom import Visdom
import os
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.c1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0)
        self.m1=nn.MaxPool2d(2,2)
        self.c2=nn.Conv2d(6,16,5,1,0)
        self.m2=nn.MaxPool2d(2,2)
        self.flat=nn.Flatten()
        self.l1=nn.Linear(16*5*5,120)
        self.a1=nn.ReLU()
        self.l2=nn.Linear(120,84)
        self.a2=nn.ReLU()
        self.l3=nn.Linear(84,10)
        # self.soft=nn.Softmax(dim=1)
    def forward(self,inputs):
        inputs=self.c1(inputs)
        inputs=self.m1(inputs)
        inputs=self.c2(inputs)
        inputs=self.m2(inputs)
        inputs=self.flat(inputs)
        inputs=self.l1(inputs)
        inputs=self.a1(inputs)
        inputs=self.l2(inputs)
        inputs=self.a2(inputs)
        outputs=self.l3(inputs)
        # outputs=self.soft(inputs)
        return outputs
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
device=torch.device('cuda')
model=Lenet().to(device)
criteon=nn.CrossEntropyLoss().to(device)
viz = Visdom(env='lenet')
optimize=torch.optim.Adam(model.parameters(),lr=1e-3)

checkpoint_save_path='./checkpoint/lenet.pkl'
if os.path.exists(checkpoint_save_path):
    print('........load model........')
    model.load_state_dict(torch.load(checkpoint_save_path))

for epoch in range(1000):
    model.train()
    for batch_idx,(x,y) in  enumerate(cifar_load):
        '''
        1.网络前向
        2.计算损失函数
        3.梯度清零
        4.loss.backward()
        5.梯度计算
        '''
        x,y=x.to(device),y.to(device)

        logits=model(x)
        loss=criteon(logits,y)#loss是一个标量
        optimize.zero_grad()#optimize梯度清零
        loss.backward()#梯度计算
        optimize.step()#梯度更新，写进optimize
    print('epoch',epoch,';loss:',loss.item())
    # viz.line([loss.item()],[epoch],win='loss',opts={'title':'train loss'},update='append')
    torch.save(model.state_dict(),checkpoint_save_path)#保存模型参数
    #test
    model.eval()
    # with torch.no_grad():#不需要计算图,可加可不加
    model.load_state_dict(torch.load(checkpoint_save_path))
    totle_correct,totle_num=0,0
    for x,y in cifar_load_test:
        x, y = x.to(device), y.to(device)
        logits=model(x)
        pred=logits.argmax(dim=1)
        totle_correct+=torch.eq(pred,y).float().sum().item()
        totle_num+=x.size(0)
    acc=totle_correct/totle_num
    # viz.line([acc],[epoch],win='acc',opts={'title':'test acc'},update='append')
    print('epoch', epoch, ';acc:', acc)