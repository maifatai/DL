import torch
from torch.nn import functional as f
'''单层感知机'''
x=torch.rand([1,10])
w=torch.randn(1,10,requires_grad=True)
o=torch.sigmoid(x@w.t())
y=torch.ones(1,1)
loss=f.mse_loss(y,o)
loss.backward()
print(w.grad)
'''MLP'''
x1=torch.rand([1,10])
w1=torch.randn(2,10,requires_grad=True)
o1=torch.sigmoid(x1@w1.t())
y1=torch.ones(1,2)
loss1=f.mse_loss(y1,o1)
loss1.backward()
print(w1.grad)