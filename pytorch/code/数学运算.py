import torch
a=torch.rand(3,4)
b=torch.rand(3,4)

c=a+b
c1=torch.add(a,b)

c2=torch.all(torch.eq(a,b))
c3=torch.all(torch.eq(a*b,torch.mul(a,b)))
c4=torch.all(torch.eq(a/b,torch.div(a,b)))

d=torch.eye(3)
e=torch.full([3,3],4.0)
res=torch.mm(d,e)#仅适用二维
res1=torch.matmul(d,e)
res2=d@e

a3 = torch.ones([4,3,2])
b3 = torch.full([4,2,4],3.)
c9=a3@b3#后两维进行矩阵乘法
c10=torch.matmul(a3,b3)
print(c9,'\n',c10)

res3=torch.pow(e,3)
res4=e.pow(3)
res5=e.sqrt()
res6=torch.sqrt(e)

res7=e.rsqrt()#平方根倒数

res8=torch.exp(e)

res9=torch.log(a)

f=torch.tensor(3.14)
print(f.floor(),'\n',f.ceil(),'\n',f.trunc(),'\n',f.frac(),'\n',f.round())
floor=torch.floor(f)
ceil=torch.ceil(f)
trunc=torch.trunc(f)#裁剪，整数部分
frac=torch.frac(f)#小数部分
round1=torch.round(f)

data=torch.rand(2,3)*15
max=data.max()
min=data.min()
median=data.median()
clap=data.clamp(10)#限幅值，小于10的值为10


