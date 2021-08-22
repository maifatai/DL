import torch
from torch import nn

rnn=nn.RNN(input_size=10,hidden_size=4,num_layers=1)#10表示Word dim，4表示memory
'''
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers
'''
W_ih=list(rnn.named_parameters())[0]#shape=4,10
W_ih_shape=rnn.weight_ih_l0.shape
W_hh=list(rnn.named_parameters())[1]#shape=4,4
W_hh_shape=rnn.weight_hh_l0.shape
b_ih=list(rnn.named_parameters())[2]#shape=4
b_ih_shape=rnn.bias_ih_l0.shape
b_hh=list(rnn.named_parameters())[3]#shape=4
b_hh_shape=rnn.bias_hh_l0.shape
dic=dict(rnn.named_parameters()).items()

x=torch.randn(5,3,10)
out,h=rnn(x,torch.zeros(1,3,4))

#4 layer
rnn1=nn.RNN(input_size=100,hidden_size=20,num_layers=4)#4层
x1=torch.randn(10,3,100)
out1,h1=rnn1(x1)

#一层
cell1=nn.RNNCell(input_size=100,hidden_size=20)
h1=torch.zeros(3,20)
x2=torch.randn(10,3,100)
for xt in x2:
    h1=cell1(xt,h1)
print(h1.shape)

#二层
cell2=nn.RNNCell(input_size=100,hidden_size=30)
cell3=nn.RNNCell(input_size=30,hidden_size=20)
h2=torch.zeros(3,30)
h3=torch.zeros(3,20)
x3=torch.randn(5,3,100)
for xt in x3:
    h2=cell2(xt,h2)
    h3=cell3(h2,h3)
