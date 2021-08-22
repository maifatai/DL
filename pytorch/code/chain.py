import torch
x=torch.tensor(1.)
w1=torch.tensor(2.,requires_grad=True)
b1=torch.tensor(1.,requires_grad=True)
w2=torch.tensor(2.,requires_grad=True)
b2=torch.tensor(1.,requires_grad=True)

hidden=x*w1+b1
out=hidden*w2+b2

d_out_hidden=torch.autograd.grad(out,[hidden],retain_graph=True)[0]
d_out_w2=torch.autograd.grad(out,[w2],retain_graph=True)[0]
d_hidden_w1=torch.autograd.grad(hidden,[w1],retain_graph=True)[0]
d_hidden_b1=torch.autograd.grad(hidden,[b1],retain_graph=True)[0]