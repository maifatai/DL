import torch

a=torch.rand(4,3,28,28)
idx1=a[0]
idx2=a[0,0]
idx3=a[0,0,2,4]


slice=a[:2]
slice1=a[:2,:1,:,:]
slice2=a[:,:,0:28:2,::2]

slice3=a[0,...]
slice4=a[:,2,...]

mask=a.ge(0.5)#将大于0.5的置为1
data=torch.masked_select(a,mask)#取出对应索引位置的数据
#上面两句等价于下面一句
data1=a[a>0.5]

b=torch.reshape(torch.arange(0,6),[2,3])
res=torch.take(b,index=torch.tensor([0,2,4]))#先将b展平，然后去索引为0,2,4对应的元素

