import tensorflow as tf

'''
tf.nn.conv2d_transpose函数来计算反卷积

value：4维的tensor，float类型，需要进行反卷积的矩阵
filter：卷积核，参数格式[height，width，output_channels，in_channels]，这里需要注意output_channels和in_channels的顺序
output_shape：一维的Tensor，设置反卷积输出矩阵的shape
strides：反卷积的步长
padding："SAME"和"VALID"两种模式
data_format：和之前卷积参数一样
name：操作的名称


注意的是，通过反卷积并不能还原卷积之前的矩阵，只能从大小上进行还原，
反卷积的本质还是卷积，只是在进行卷积之前，会进行一个自动的padding补0，
从而使得输出的矩阵与指定输出矩阵的shape相同。

在进行反卷积的时候设置的stride并不是指反卷积在进行卷积时候卷积核的移动步长，
而是被卷积矩阵填充的padding

反卷积的应用
CNN可视化，通过反卷积将卷积得到的feature map还原到像素空间，来观察feature map对哪些pattern相应最大，即可视化哪些特征是卷积操作提取出来的；
FCN全卷积网络中，由于要对图像进行像素级的分割，需要将图像尺寸还原到原来的大小，类似upsampling的操作，所以需要采用反卷积；
GAN对抗式生成网络中，由于需要从输入图像到生成图像，自然需要将提取的特征图还原到和原图同样尺寸的大小，即也需要反卷积操作。


'''
if __name__ == "__main__":
    x1 = tf.constant([4.5,5.4,8.1,9.0],shape=[1,2,2,1],dtype=tf.float32)
    dev_con1 = tf.ones(shape=[3,3,1,1],dtype=tf.float32)
    y1 = tf.nn.conv2d_transpose(x1,dev_con1,output_shape=[1,4,4,1],strides=[1,1,1,1],padding="VALID")

    print(y1)
    print(x1)
