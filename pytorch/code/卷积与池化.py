import torch
from torch.nn import functional as F
#类方法
inputs=torch.rand(10,3,28,28)
conv1=torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=0)#无填充
conv2=torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)#填充

max_pooling=torch.nn.MaxPool2d(kernel_size=2,stride=2)
avg_pooling=torch.nn.AvgPool2d(kernel_size=2,stride=2)
out1=conv1.forward(inputs)#不推荐使用
out11=conv1(inputs)#自动调用
out12=max_pooling(out11)

out2=conv2.forward(inputs)

conv1_weight=conv1.weight
conv1_weight_shape=conv1_weight.shape
conv1_bias=conv1.bias

out1_shape=out1.shape
out2_shape=out2.shape

#函数方法

out3=F.conv2d(input=inputs,weight=torch.rand(16,3,3,3),bias=torch.rand(16),stride=1,padding=1)
out31=F.avg_pool2d(out3,2,stride=2)
out32=F.max_pool2d(out3,2,stride=2)

#上采样
upsampling=F.interpolate(out32,scale_factor=2,mode='nearest')