import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a=tf.random.shuffle(tf.range(10))
res1=tf.sort(a,direction="DESCENDING")
#默认是升序，降序修改参数direction
print(res1)
res2=tf.argsort(a,direction='DESCENDING')
#返回排序结果的索引
res3=tf.gather(a,res2)

'''
二维
'''
a1=tf.random.normal([3,3])
res4=tf.sort(a1)
#默认在最后一个维度进行排序，二维为axis=1，升序
print(res4)
idx=tf.argsort(a)
print(idx)

'''
最大的前几个数的排序tf.math.top_k()
适用在精度的确认时，top5
'''
b=tf.reshape(a,[2,-1])
res5=tf.math.top_k(b,3)#最大的前三个数
print(res5.indices)#返回索引值
print(res5.values)#返回值

prob=tf.constant([[.1,.2,.7],[.2,.7,.1]])
target=tf.constant([2,0])
k_b=tf.math.top_k(prob,3).indices
k_b=tf.transpose(k_b,[1,0])
target=tf.broadcast_to(target,[3,2])

'''
top-k accuracy
'''
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)
    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k* (100.0 / batch_size) )
        res.append(acc)
    return res

output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob:', output.numpy())
pred = tf.argmax(output, axis=1)
print('pred:', pred.numpy())
print('label:', target.numpy())

acc = accuracy(output, target, topk=(1,2,3,4,5,6))
print('top-1-6 acc:', acc)