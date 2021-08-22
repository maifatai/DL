import torch
from torch import nn

#类似于nn.RNN

lstm=nn.LSTM(input_size=10,hidden_size=4,num_layers=1)#10表示Word dim，4表示memory
'''
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers
'''
W_ih=list(lstm.named_parameters())[0]#shape=4,10
W_ih_shape=lstm.weight_ih_l0.shape
W_hh=list(lstm.named_parameters())[1]#shape=4,4
W_hh_shape=lstm.weight_hh_l0.shape
b_ih=list(lstm.named_parameters())[2]#shape=4
b_ih_shape=lstm.bias_ih_l0.shape
b_hh=list(lstm.named_parameters())[3]#shape=4
b_hh_shape=lstm.bias_hh_l0.shape
dic=dict(lstm.named_parameters()).items()

x=torch.randn(5,3,10)
out,(ht,ct)=lstm(x)

#4 layer
lstm1=nn.LSTM(input_size=100,hidden_size=20,num_layers=4)#4层
x1=torch.randn(10,3,100)
out1,(h1,c1)=lstm1(x1)


#一层
cell1=nn.LSTMCell(input_size=100,hidden_size=20)
h1=torch.zeros(3,20)
c1=torch.zeros(3,20)
x2=torch.randn(10,3,100)
for xt in x2:
    h1,c1=cell1(xt,[h1,c1])
print(h1.shape)

#二层
cell2=nn.LSTMCell(input_size=100,hidden_size=30)
cell3=nn.LSTMCell(input_size=30,hidden_size=20)
h2=torch.zeros(3,30)
c2=torch.zeros(3,30)
h3=torch.zeros(3,20)
c3=torch.zeros(3,20)
x3=torch.randn(5,3,100)
for xt in x3:
    h2,c2=cell2(xt,[h2,c2])
    h3,c3=cell3(h2,[h3,c3])