import torch
import torch.nn as nn
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
net=Lenet()
input=torch.randn(2,3,32,32)
out=net(input)
print(out.shape)