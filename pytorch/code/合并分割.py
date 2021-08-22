import torch
a=torch.rand(4,32,8)
b=torch.rand(5,32,8)
c=torch.rand(4,32,8)
#合并
res=torch.cat([a,b,c],dim=0)#在第0轴合并
'''stack创建新的一个维度'''
res1=torch.stack([a,c],dim=0)
res2=torch.stack([a,c],dim=3)

#拆分
res3=a.split([1,1,2],dim=0)
res4=a.chunk(4,dim=1)#均分成4部分