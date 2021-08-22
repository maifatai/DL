import torch

# net=torch.nn.Sequential(
#     torch.nn.Linear(784,200),
#     torch.nn.BatchNorm1d(),
#     torch.nn.Dropout(0.5),
#     torch.nn.ReLU()
# )
#torch.nn.Dropout(0.3)表示丢弃的概率为0.3
#tf.nn.dropout(0.3)表示保留的概率为0.3

#在测试时，要将dropout改为全连接。

#BN
x=torch.rand(100,16,784)#图片数量，通道数，像素数28*28=784
layer=torch.nn.BatchNorm1d(16)#记录每个channel上的均值和方差
out=layer(x)
mean=layer.running_mean
var=layer.running_var

x1=torch.rand(10,3,28,28)
bn2=torch.nn.BatchNorm2d(3)
out2=bn2(x1)

weight=bn2.weight
bias=bn2.bias

#test时，mean和var时全局的，training时，是batch