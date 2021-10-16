import torch
from torch import nn

'''
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                          stride=1, padding=0, output_padding=0,
                          groups=1, bias=True, dilation=1, padding_mode='zeros')
                          
in_channels(int)：输入张量的通道数
out_channels(int)：输出张量的通道数
kernel_size(int or tuple)：卷积核大小
stride(int or tuple,optional)：卷积步长，决定上采样的倍数
padding(int or tuple, optional)：对输入图像进行padding，输入图像尺寸增加2*padding
output_padding(int or tuple, optional)：对输出图像进行padding，输出图像尺寸增加padding
groups：分组卷积（必须能够整除in_channels和out_channels）
bias：是否加上偏置
dilation：卷积核之间的采样距离（即空洞卷积）
padding_mode(str)：padding的类型
另外，对于可以传入tuple的参数，tuple[0]是在height维度上，tuple[1]是在width维度上

输入输出尺寸的公式
output=(input-1)*stride+outputpadding-2*padding+kernalsize
'''
tmp=torch.randn(2,3,224,224)
convt=nn.ConvTranspose2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1,output_padding=1)
out=convt(tmp)
print(out.shape)
