import tensorflow as tf
'''
强制tensor转换为该数据类型 
tf.cast (张量名，dtype=数据类型) 
计算张量维度上元素的最小值 
tf.reduce_min (张量名) 
计算张量维度上元素的最大值 
tf.reduce_max (张量名)
'''
x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)
x2 = tf.cast(x1, tf.int32)
print("x2", x2)
print("minimum of x2：", tf.reduce_min(x2))
print("maxmum of x2:", tf.reduce_max(x2))
'''
在一个二维张量或数组中，可以通过调整 axis 等于0或1 控制执行维度。 􏰀 
axis=0代表跨行(经度，down)，而axis=1代表跨列(纬度，across) 􏰀 
如果不指定axis，则所有元素参与计算。
'''
x=tf.random.normal([5,3],mean=0,stddev=1)
axis_0=tf.reduce_max(x,axis=0)
axis_1=tf.reduce_min(x,axis=1)
mean0=tf.reduce_mean(x,axis=0)
mean1=tf.reduce_mean(x,axis=1)
mean=tf.reduce_mean(x)
argmax=tf.argmax(x)
argmax0=tf.argmax(x,axis=0)
argmax1=tf.argmax(x,axis=1)
argmin=tf.argmin(x)
argmin0=tf.argmin(x,axis=0)
argmin1=tf.argmin(x,axis=1)
print(axis_0)
print(axis_1)
'''
向量的范数
'''
a=tf.ones([2,2])
norm=tf.norm(a)#二范数
norm1=tf.sqrt(tf.reduce_sum(tf.square(a)))
a1=tf.ones([4,28,28,3])
norm2=tf.norm(a1)
norm3=tf.sqrt(tf.reduce_sum(tf.square(a1)))

norm4=tf.norm(a,ord=1)#一范数
norm5=tf.norm(a,ord=1,axis=0)#一范数,在第0轴
norm6=tf.norm(a,ord=1,axis=1)#一范数，在第1轴

'''
比较,利用argmax可以作为预测的值得索引，然后与真实值进行tf.equal，然后通过tf.reduce_sum(tf.cast)计算准确率
'''
a=tf.constant([1,2,3,2,5])
b=tf.range(5)
equal=tf.equal(a,b)
num=tf.reduce_sum(tf.cast(equal,dtype=tf.int32))

'''tf.unique计算不重复的元素'''
c=tf.constant([1,2,3,4,2,3])
res1=tf.unique(c)
res2=tf.gather(res1[0],res1[1])#还原c