import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from  torchvision import datasets, transforms

class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.act1=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.act2=nn.ReLU()
        self.extra=nn.Sequential()
        if out_channel!=in_channel:
            self.extra=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self,inputs):
        out=self.conv1(inputs)
        out=self.bn1(out)
        out=self.act1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.act2(out)
        out=self.extra(inputs)+out
        return out
