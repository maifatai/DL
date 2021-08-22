import torch

#L1 正则化

regularization_loss=0
for param in model.parameters():
    regularization_loss+=torch.sum(torch.abs(param))
classify_loss=critron(logits,target)
loss=classify_loss+0.01*regularization_loss