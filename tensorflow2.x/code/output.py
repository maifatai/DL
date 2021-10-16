import tensorflow as tf
'''sigmoid函数输出为[0,1]'''
a=tf.linspace(-6,6,10)
sigmoid=tf.sigmoid(a)
print(sigmoid)

'''softmax输出的概率之和为1'''

softmax=tf.nn.softmax(a)
print(softmax)

tanh=tf.tanh(a)
print(tanh)

'''
误差计算
1.MES
2.交叉熵:两个分部之间的关系
3.Hinge loss
'''
y=tf.constant([1,2,3,0,2])
y=tf.one_hot(y,depth=4)
y=tf.cast(y,dtype=tf.float32)

out=tf.random.normal([5,4])

loss1=tf.reduce_mean(tf.square(y-out))

loss2=tf.square(tf.norm(y-out))

loss3=tf.reduce_mean(tf.losses.MSE(y,out))
print('loss1',loss1)
print('loss2',loss2)
print('loss3',loss3)

#交叉熵
#函数形式
cross_entropy=tf.losses.categorical_crossentropy([0,1,0,0],[0.25,.25,.25,.25])
cross_entropy1=tf.losses.categorical_crossentropy([0,1,0,0],[0,0.98,.01,.01])
binary_cross_entropy1=tf.losses.binary_crossentropy([1],[0.1])
#类形式
cross_entropy2=tf.losses.CategoricalCrossentropy()([0,1,0,0],[0,0.98,.01,.01])
binary_cross_entropy=tf.losses.BinaryCrossentropy()([1],[0.1])

#损失函数及其梯度

x=tf.random.normal([2,4])
w=tf.random.normal([4,3])
b=tf.zeros([3])
y=tf.constant([2,0])
with tf.GradientTape() as tape:
    tape.watch([w,b])
    logits=x@w+b
    loss=tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y,depth=3),logits,from_logits=True))
grads=tape.gradient(loss,[w,b])