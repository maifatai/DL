import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a=tf.range(10)
maxmum=tf.maximum(a,5)#大于等于5返回原值，小于5返回5，类似于rulu函数
minmum=tf.minimum(a,5)
clip_by_value=tf.clip_by_value(a,2,8)#在2-8的范围内取原值，除此之外取两端的值
b=a-5
relu=tf.nn.relu(b)

c=tf.random.normal([2,2],mean=10)
clip_by_norm=tf.clip_by_norm(c,15)#范数等比例缩放，最大值为15

# grads, _ = tf.clip_by_global_norm(grads, 15)  # 等比例缩放，可以防止梯度爆炸和梯度消失


