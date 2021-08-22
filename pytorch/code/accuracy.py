import torch
from torch.nn import functional as F
logits=torch.rand(4,10)
logits_label=logits.argmax(dim=1)
pred=F.softmax(logits,dim=1)
pred_label=pred.argmax(dim=1)
label=torch.tensor([7,2,1,4])

correct=torch.eq(pred_label,label)

accuracy=correct.sum().float().item()/len(label)#item()是将数据转化numpy数据