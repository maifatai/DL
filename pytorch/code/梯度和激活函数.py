import torch
from torch.nn import functional as f
'''激活函数'''
a=torch.linspace(-100,100,10)
sigmoid=torch.sigmoid(a)
sigmoid1=f.sigmoid(a)
tanh=torch.tanh(a)
tanh1=f.tanh(a)
relu=torch.relu(a)
relu1=f.relu(a)

ones=torch.ones(4)
full=torch.full([4],3)
norm=torch.norm(full-ones)


x=torch.ones(1)
w=torch.full([1],2.0,requires_grad=True)#需要梯度信息
b=torch.zeros(1,requires_grad=True)
mse=f.mse_loss(torch.ones(1),x*w)
'''两种求梯度的方法'''
# grad=torch.autograd.grad(mse,[w])
mse=f.mse_loss(torch.ones(1),x*w+b)
#
mse.backward()#反向传播求导，将梯度信息加到tensor的成员变量grad中

print(w.grad,b.grad)

a=torch.rand(3)
a.requires_grad()
p=f.softmax(a,dim=0)
p.backward()

