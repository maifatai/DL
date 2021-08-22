import torch

a=torch.rand(4,3,28,28)
#view和reshape一样
res=a.view(4,3,28*28)
res1=torch.reshape(a,[4,3,28*28])



#维度缩减和扩展,类似TensorFlow
#维度增加
b=torch.rand(4,1,28,28)
b1=b.unsqueeze(0)#参数为轴
b2=b.unsqueeze(-5)

b3=b.unsqueeze(4)
b4=b.unsqueeze(-1)

c=torch.tensor([2,1.0])#一维
c1=c.unsqueeze(-1)#二维，2*1
c2=c.unsqueeze(0)#二维，1*2

d=torch.rand(32)
d1=d.unsqueeze(1).unsqueeze(2).unsqueeze(0)

#维度减少
d2=d.squeeze()
d3=d.squeeze(0)
d4=d.squeeze(-1)
#区别维度增加和维度扩展
#维度扩展 [1,32,1,1]->[4,32,28,28]
d5=d1.expand(4,32,28,28)
d7=d1.expand(-1,32,4,-1)#-1表示原来的维度不进行扩张

d6=d1.repeat(4,1,28,28)#参数表示扩展的倍数

#转置
e=torch.rand(3,4)
e1=e.t()

#维度交换
e2=d5.transpose(1,3)#参数为交换的维度
e3=d5.transpose(1,3).contiguous().view(4,32*28*28).view(4,28,28,32).transpose(1,3)#contiguous()使得内存连续
e4=torch.all(torch.eq(d5,e3))

e5=d5.permute(0,2,3,1)