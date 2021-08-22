import torch
from torch import nn
from torchnlp.word_to_vector import GloVe
word={'hello':0,'world':1}
lookup=torch.tensor([word['hello']],dtype=torch.long)

embed=nn.Embedding(2,5)#embedding 编码方式
hell0_embed=embed(lookup)
print(hell0_embed)

vector=GloVe()
print(vector['hello'])