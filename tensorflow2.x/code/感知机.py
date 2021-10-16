import tensorflow as tf
#单输出，单层感知机
x=tf.random.normal([1,4])
w=tf.random.normal([4,1])
b=tf.ones([1])
y=tf.constant([1])
with tf.GradientTape() as tape:
    tape.watch([w,b])
    logits=x@w+b
    activation=tf.sigmoid(logits)
    loss=tf.reduce_mean(tf.losses.MSE(y,activation))
grads=tape.gradient(loss,[w,b])

#多输出，单层感知机
x=tf.random.normal([2,4])
w=tf.random.normal([4,3])
b=tf.zeros([3])
y=tf.constant([2,0])
with tf.GradientTape() as tape:
    tape.watch([w,b])
    logits=tf.nn.softmax(x@w+b,axis=1)
    loss=tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth=3),logits))
grads=tape.gradient(loss,[w,b])

#多层感知机


