from sklearn import datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''step1：准备数据'''
'''导入数据集'''
x_data = datasets.load_iris().data  # .data返回iris数据集所有输入特征
y_data = datasets.load_iris().target  # .target返回iris数据集所有标签
"""打乱数据集"""
seed=110
np.random.seed(seed)
np.random.shuffle(x_data)
np.random.seed(seed)
np.random.shuffle(y_data)
tf.random.set_seed(seed)
'''120个训练集，30个测试集'''
x_training=x_data[:120]
y_training=y_data[:120]
x_testing=x_data[120:]
y_testing=y_data[120:]
x_training=tf.cast(x_training,tf.float32)
x_testing=tf.cast(x_testing,tf.float32)
'''配对特征和标签'''
batch=30
training=tf.data.Dataset.from_tensor_slices((x_training,y_training)).batch(batch)
testing=tf.data.Dataset.from_tensor_slices((x_testing,y_testing)).batch(batch)

'''step2：搭建网络'''
#三层网络，其中一层隐含层
hidden_nn=5#隐藏层神经元个数
w1=tf.Variable(tf.random.normal([4,hidden_nn],mean=0,stddev=1.0))
b1=tf.Variable(tf.random.normal([hidden_nn],mean=0,stddev=1.0))
w2=tf.Variable(tf.random.normal([hidden_nn,3],mean=0,stddev=1.0))
b2=tf.Variable(tf.random.normal([3],mean=0,stddev=1.0))

"""step3：参数优化"""
lr=0.1#学习lv
training_loss=[]#训练损失
testing_acc=[]#测试准确率
epoch=10000#训练轮数
loss_all=0
classes=3#分类的类别数

for epoch in range(epoch):
    for step,(x_training,y_training) in enumerate(training):#batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:# with结构记录梯度信息
            hidden=(tf.matmul(x_training,w1)+b1)
            hidden=tf.nn.relu(hidden)
            y=tf.matmul(hidden,w2)+b2
            y=tf.nn.softmax(y)
            y_=tf.one_hot(y_training,depth=classes)
            loss=tf.reduce_mean(tf.square(y_-y))# 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all+=loss.numpy()# 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        training_variables=[w1,b1,w2,b2]
        grad=tape.gradient(loss,training_variables)
        w1.assign_sub(lr*grad[0])#原地更新
        b1.assign_sub(lr*grad[1])
        w2.assign_sub(lr*grad[2])
        b2.assign_sub(lr*grad[3])
    # print("epoch:{},loss:{}".format(epoch,loss_all/4))
    training_loss.append(loss_all/4)
    loss_all=0# loss_all归零，为记录下一个epoch的loss做准备

    """step4:测试效果"""
    totle_correct,totle_num=0,0
    for (x_testing,y_testing) in testing:
        y=tf.matmul((tf.matmul(x_testing,w1)+b1),w2)+b2
        y=tf.nn.softmax(y)
        predict=tf.argmax(y,axis=1)# 返回y中最大值的索引，即预测的分类
        predict=tf.cast(predict,dtype=y_testing.dtype)
        correct=tf.cast(tf.equal(predict,y_testing),dtype=tf.int32)
        correct=tf.reduce_sum(correct)
        totle_correct+=int(correct)
        totle_num+=x_testing.shape[0]
    acc=totle_correct/totle_num
    testing_acc.append(acc)
    # print("test acc:",acc)
""" step5：acc/loss曲线"""
plt.title('loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(training_loss,label="$loss$")
plt.show()

plt.title('acc curve')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.plot(testing_acc,label="$accuracy$")
plt.show()