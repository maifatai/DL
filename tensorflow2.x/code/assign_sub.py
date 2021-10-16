import tensorflow as tf
"""
assign_sub
tf.assign_sub
 赋值操作，更新参数的值并返回。
 调用assign_sub前，先用 tf.Variable 定义变量 w 为可训练(可自更新)。
w.assign_sub (w要自减的内容)
"""
x = tf.Variable(4)
x.assign_sub(1)
print("x:", x)  # 4-1=3
