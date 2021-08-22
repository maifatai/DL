import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
def main():
    cifar_train=datasets.CIFAR10('cifar10',#在当前目录下
                           train=True,
                           transform=transforms.Compose(
                               [transforms.Resize((32,32)),
                                transforms.ToTensor()]
                           ),download=True)#一次只能加载一个
    cifar_load=DataLoader(dataset=cifar_train,batch_size=32,shuffle=True)
    cifar_test=datasets.CIFAR10('cifar10',#在当前目录下
                           train=False,
                           transform=transforms.Compose(
                               [transforms.Resize((32,32)),
                                transforms.ToTensor()]
                           ),download=True)#一次只能加载一个
    cifar_load_test=DataLoader(dataset=cifar_test,batch_size=32,shuffle=True)

    x,label=iter(cifar_load).next()
    print(x.shape,label.shape)
if __name__=='__main__':
    main()