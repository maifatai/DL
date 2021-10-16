import tensorflow as tf

a=tf.random.normal([3,3])
sigmoid=tf.sigmoid(a)
sigmoid1=tf.nn.sigmoid(a)

# with tf.GradientTape() as tape:
#     tape.watch(a)
#     sigmoid=tf.sigmoid(a)
#     grad=tf.gradients(sigmoid,a)

tanh=tf.tanh(a)#在RNN中使用较多
tanh1=tf.nn.tanh(a)
relu=tf.nn.relu(a)
leak_relu=tf.nn.leaky_relu(a)

softmax=tf.nn.softmax(a)