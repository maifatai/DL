import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
tf.where于numpy类似
'''
a=tf.random.normal([3,3])
mask=a>0
bool_mask=tf.boolean_mask(a,mask)
#返回对应位置为TRUE的元素
indices=tf.where(mask)#返回对应的索引

res=tf.gather_nd(a,indices)
#结果与bool_mask=tf.boolean_mask(a,mask)类似

res1=tf.where(a>5,a-5,a)#返回对应的值
print(res1)

'''
scatter_nd(indices, updates, shape, name=None):
适用于指定位置的更新或者指定位置的相加减等运算
'''
indices=tf.constant([[4],[3],[1],[7]])
updates=tf.constant([9,10,11,12])
shape=tf.constant([8])#
res2=tf.scatter_nd(indices,updates,shape)
#indices表示要更新的索引，updates表示对应索引位置更新的后的值
#底板为一个shape=8，即长度为8，[0,0,0,0,0,0,0,0],且全为0。

#高维
indices1=tf.constant([[0],[2]])
shape1=tf.constant([4,4,4])
updates1=tf.reshape(tf.range(1,33),[2,4,-1])
res3=tf.scatter_nd(indices1,updates1,shape1)
print(res3)

b=tf.linspace(-2,2,5)
print(b)
c=tf.linspace(-2,2,5)
x,y=tf.meshgrid(b,c)#生成坐标网格点
points=tf.stack([x,y],axis=2)

def func(x):
    """
    :param x: [b, 2]
    :return:
    """
    z = tf.math.sin(x[...,0]) + tf.math.sin(x[...,1])
    return z

x = tf.linspace(0., 2*3.14, 500)
y = tf.linspace(0., 2*3.14, 500)
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y], axis=2)
# points = tf.reshape(points, [-1, 2])
print('points:', points.shape)
z = func(points)
print('z:', z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()