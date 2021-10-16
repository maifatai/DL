import tensorflow as tf
'''
with结构记录计算过程，gradient求出张量的梯度
with tf.GradientTape( ) as tape: 
    若干个计算过程
grad=tape.gradient(函数，对谁求导)
'''
with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))
    y = tf.pow(x, 2)
grad = tape.gradient(y, x)
print(grad)
#二阶梯度
with tf.GradientTape() as tape:
    with tf.GradientTape() as tape1:
        a=tf.Variable(tf.constant(3.0))
        b=tf.Variable(tf.constant(3.0))
        y=tf.pow(a,2)+tf.pow(b,2)
    y_a,y_b=tape1.gradient(y,[a,b])
y_aa=tape.gradient(y_a,a)#二阶梯度


