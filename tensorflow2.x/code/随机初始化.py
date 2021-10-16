import tensorflow as tf
'''
如何创建一个Tensor 􏰀生成正态分布的随机数，默认均值为0，标准差为1
tf. random.normal (维度，mean=均值，stddev=标准差) 􏰀生成截断式正态分布的随机数
tf. random.truncated_normal (维度，mean=均值，stddev=标准差)
在tf.truncated_normal中如果随机生成数据的取值在(μ-2σ，μ+2σ)之外 
则重新进行生成，保证了生成值在均值附近。
μ:均值， σ:标准差
'''
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)

'''
生成均匀分布随机数 [ minval, maxval )
tf. random. uniform(维度，minval=最小值，maxval=最大值)
'''
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)
