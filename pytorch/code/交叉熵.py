import torch
from torch.nn import functional as f
a=torch.full([4],.25)
entropy=-(a*torch.log2(a)).sum()

x=torch.randn(1,784)
w=torch.randn(10,784)
'''CE=softmax+log+nll_loss'''
logits=x@w.t()
pred=f.softmax(logits,dim=1)
pred_log=torch.log(pred)
cross_entropy1=f.nll_loss(pred_log,torch.tensor([3]))


cross_entropy=f.cross_entropy(logits,torch.tensor([3]))

