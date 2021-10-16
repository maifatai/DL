import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x=tf.random.normal([4,784])
net=tf.keras.layers.Dense(512)
y=net(x)#自动创建bulid
print(y.shape)#输出的维度
print(net.kernel.shape,net.bias.shape)#网络的维度
print('net get-weight：',net.get_weights(),'\n',net.weights,'\n',net.bias)

net1=tf.keras.layers.Dense(10)
print('net1 weight',net1.get_weights())
print(net1.weights)
net1.build(input_shape=(None,4))#bulid用来创建权重和偏置项
print(net1.kernel.shape,net1.bias.shape)
print(net1.kernel,net1.bias)

net1.build(input_shape=(None,20))
print(net1.kernel.shape,net1.bias.shape)
print(net1.kernel,net1.bias)

