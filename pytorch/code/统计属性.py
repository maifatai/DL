import torch
a=torch.rand(2,3)
norm1=a.norm(1)
norm2=a.norm(2)

res=a.norm(1,dim=1)
#可以在某个轴上进行相应的方法dim      keepdim(返回的和原来的维度一样)
mean=a.mean()
prod=a.prod()#累乘
argmin=a.argmin()#展平之后的索引
argmax=a.argmax()
argmax_dim1=a.argmax(dim=1)
argmax_keepdim1=a.argmax(dim=1,keepdim=True)


#top-k
b=torch.rand(4,10)
topk=b.topk(2,dim=1)#最大的k个
topk1=b.topk(3,dim=1,largest=False)#最小的k个

kvalue=a.kthvalue(8,dim=1)#在第1轴上第8个小值

#比价运算
'''
'''
torch.eq()#比较两个tensor的相对应位置
torch.ge()
torch.le()
torch.equal()#比较两个tensor的所有元素