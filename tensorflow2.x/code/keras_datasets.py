import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''
keras 数据集  keras.datasets
1.Boston housing   波士顿房价
2.mnist/fashion mnist
3.cifar10/100    small images classification datasets
4.imdb    sentiment classificationdatasets
'''
(training_x,training_y),(test_x,test_y)=keras.datasets.mnist.load_data()
print(training_x.shape,training_y.shape,test_x.shape,test_y.shape)
y_onehot=tf.one_hot(training_y,depth=10)

(training_x1,training_y1),(test_x1,test_y1)=keras.datasets.cifar10.load_data()
print(training_x1.shape,training_y1.shape,test_x1.shape,test_y1.shape)

(training_x2,training_y2),(test_x2,test_y2)=keras.datasets.cifar100.load_data()
print(training_x2.shape,training_y2.shape,test_x2.shape,test_y2.shape)

db=tf.data.Dataset.from_tensor_slices((training_x2,training_y2))
db=db.shuffle(10000)#打散
db3=db.batch(32)
print(next(iter(db))[0].shape)

#数据预处理
'''
将数据转换到0-1之间，并且转换为浮点类型，讲标签转化为one-hot类型
'''

'''
tf.data.Dataset
生成特征标签对
切分传入张量的第一维度，生成输入特征/标签对，构建数据集 data = tf.data.Dataset.from_tensor_slices((输入特征, 标签))
(Numpy和Tensor格式都可用该语句读入数据)
'''
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)
#print(next(iter(dataset)))