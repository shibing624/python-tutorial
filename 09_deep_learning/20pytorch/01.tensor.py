# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
import torch

x = torch.Tensor(5, 2)
print(x)

y = torch.rand(3, 4)
print(y)
print(len(y))
print(y.size())

z = torch.rand(3, 1)
print(z)
o = torch.add(y, z)
print(o)

k = torch.Tensor(3, 4)
torch.add(y, z, out=k)
print(k)

a = torch.ones(3)
print(a)

b = torch.FloatTensor(3, 4)
print(b)

b = a.add_(2)
print(b)

import numpy as np

a = np.ones(3)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
