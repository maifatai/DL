import torch
import  torch.nn as nn
import  torch.nn.functional as F
'''
激活函数的调用
1.类方法 nn.
2.函数 F.
'''
x=torch.randn([2,3])
leak_relu=nn.LeakyReLU(x)
leak_relu1=F.leaky_relu(x)
selu=nn.SELU(x)
selu1=F.selu(x)
softplus=nn.Softplus(x)
softplus1=F.softplus(x)


