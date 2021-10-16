import tensorflow as tf

with tf.device('cpu'):
    a=tf.constant(1)#a是在CPU设备环境中创建
with tf.device('gpu'):
    b=tf.range(4)#b是在PU设备环境中创建
print(a.device)#查询变量在那个设备环境
print(b.device)