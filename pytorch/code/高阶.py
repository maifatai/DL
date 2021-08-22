import torch
a=torch.rand(3,3)*10


res1=torch.where(a>5,a-5,a)#返回对应的值
print(res1)

mask=a>5
indices=torch.where(mask)

a2=torch.rand([4,35,8])
'''tf.gather仅在某一个维度上'''
data,idx=a2.topk(dim=1,k=3)
a2_1=torch.gather(a2,dim=1,index=idx)
# a2_1=torch.gather(a2,dim=0,index=[2,3])#在axis=0的维度上
# a2_2=torch.gather(a2,dim=1,index=[2,30,3,17,6])
# a2_3=torch.gather(a2,dim=2,index=[2,3,6])