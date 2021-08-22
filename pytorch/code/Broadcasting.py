import torch

a=torch.arange(0,40,10).view(-1,1)#[4,1]
b=torch.arange(0,3).view(1,-1)#[1,3]
res=a+b#[4,3]

c=torch.ones([3,3])
d=torch.full([2,2,3,3],2)
res1=c+d