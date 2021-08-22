import torch
from visdom import Visdom
import numpy as np

# 新建名为'demo'的环境
viz = Visdom(env='demo')

arr = np.random.rand(10)

# Numpy Array
viz.line(Y=arr)
# Python List
viz.line(Y=list(arr))
# PyTorch tensor
viz.line(Y=torch.Tensor(arr))