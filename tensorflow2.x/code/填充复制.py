import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
tf.pad(tensor, paddings, mode="CONSTANT", 
constant_values=0, name=None):
常用在图像的填充（卷积），句子填充为相同的长度

"CONSTANT", "REFLECT", or "SYMMETRIC" 
'''
a=tf.ones([3,3])
print(a)
res=tf.pad(a,[[0,0],[0,0]])
res1=tf.pad(a,[[1,0],[0,0]])
#在上方填充一行，默认填充0
res2=tf.pad(a,[[1,1],[0,0]])
#在上方、下方各填充一行，默认填充0
res3=tf.pad(a,[[1,1],[2,2]])
#在上方、下方各填充一行，在左侧和右侧各填充两行，默认填充0
print(res3.shape)

'''图像的填充'''
img=tf.random.normal([4,28,28,3])
pad_img=tf.pad(img,[[0,0],[2,2],[2,2],[0,0]])

'''
复制。类似numpy的tile,结果与broadcast_to类似，
尽量使用broadcast_to
'''
tile=tf.tile(a,[2,3])
print(tile)
