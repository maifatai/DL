import tensorflow as tf
import numpy as np

"""
参考numpy
"""
a=np.arange(10).reshape(-1,1)
b=np.arange(5).reshape(1,-1)
a1=tf.convert_to_tensor(a)
b1=tf.convert_to_tensor(b)
c=a1+b1
print(c)
'''
Y=X@W+b
feature map:[4,32,32,3]
Bias:[3]
'''
X=tf.ones([4,2])
W=tf.ones([2,1])
b=tf.constant(0.1)
Y=X@W+b
out=tf.nn.relu(Y)

x=tf.random.normal([4,32,32,3])
y=x+tf.random.normal([3])
y1=x+tf.random.normal([32,32,1])
y2=x+tf.random.normal([4,1,1,1])
y3=tf.broadcast_to(
    tf.random.normal([4,1,1,1]),[4,32,32,3])

print(y.shape,y1.shape,y2.shape)