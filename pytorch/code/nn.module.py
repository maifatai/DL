import torch
import torch.nn as nn

'''
1.nn.Module中含有
nn.Linear
nn.Conv2d
nn.BatchNorm2d
nn.Dropout等，并且可以嵌套
2.含有nn.Sequential，只有类才可以写到Sequential
self.net=nn.Seqential(
    nn.Linear,
    my_linear,#自己定义的layer)
3.参数管理 
net.parameters()
4.module
modules：all nodes
children：direct children
5.将参数转移到GPU
6.保存和加载
7.train/test
对于dropout和BN在train和test中时不一样的
8.实现自己定义的layer
'''
#3.参数管理
net=nn.Sequential(nn.Linear(4,2),nn.Linear(2,2))
para=net.parameters()
para1=list(net.parameters())[0]#必须要加list，对应第一个全连接层的w
para2=list(net.parameters())[1]#对应第一个全连接层的b
para3=list(net.parameters())[2]#对应第二个全连接层的w
para4=list(net.parameters())[3]#对应第一个全连接层的b
#每层的参数都有对应的名字,net.named_parameters()返回一个字典
para11=list(net.named_parameters())[0]
para21=list(net.named_parameters())[1]
para31=list(net.named_parameters())[2]
para41=list(net.named_parameters())[3]

dic=dict(net.named_parameters()).items()

#4.module
#8.实现自己定义的layer
class MyLinear(nn.Module):
    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()
        # requires_grad = True
        self.w = nn.Parameter(torch.randn(outp, inp))
        #nn.Parameter是一个包装器，可以将参数加到可训练的参数，方便之后net.parameter的调用
        self.b = nn.Parameter(torch.randn(outp))
    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(1, 16, stride=1, padding=1),
                                 nn.MaxPool2d(2, 2),
                                 Flatten(),
                            nn.Linear(1*14*14, 10))
    def forward(self, x):
        return self.net(x)



class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.net = nn.Linear(4, 3)
    def forward(self, x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(BasicNet(),#自己定义的layer
                                 nn.ReLU(),
                                 nn.Linear(3, 2))
    def forward(self, x):
        return self.net(x)

def main():
    #5.将数据转移到GPU
    device = torch.device('cuda')
    net = Net()
    net.to(device)
    net.train()
    net.eval()

    # net.load_state_dict(torch.load('ckpt.mdl'))
    #
    #
    # torch.save(net.state_dict(), 'ckpt.mdl')

    for name, t in net.named_parameters():
        print('parameters:', name, t.shape)
    for name, m in net.named_children():
        print('children:', name, m)
    for name, m in net.named_modules():
        print('modules:', name, m)

#6.保存和加载
net.load_state_dict(torch.load('ckpt.mdl'))#加载模型网络参数
model=torch.load('ckpt.mdl')
torch.save(net.state_dict(),'ckpt.mdl')#保存模型网络参数
torch.save(net,'ckpt.mdl')#保存整个网络

#7.train/test
#train
net.train()
#test
net.eval()