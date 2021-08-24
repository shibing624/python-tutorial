# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import torch
from torch import nn


def pool2d(X, pool_size, mode="max"):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'mean':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
c = pool2d(X, (2, 2))
print(c)

# mean
c = pool2d(X, (2, 2), mode='mean')
print(c)

# other, mean
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 9]])
c = pool2d(X, (2, 2), mode='mean')
print(c)

X = torch.arange(32, dtype=torch.float).view((1, 2, 4, 4))  # view, bathsize, channels, height, width
print(X)
X = torch.arange(32, dtype=torch.float).view((2, 1, 4, 4))  # view, bathsize, channels, height, width
print(X)

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))  # view, bathsize, channels, height, width
print(X)

pool2d = nn.MaxPool2d(3)
c = pool2d(X)
print(c)

pool2d = nn.MaxPool2d((3, 3))
c = pool2d(X)
print(c)

# 多通道
X = torch.cat((X, X + 1), dim=1)
print(X)

pool2d = nn.MaxPool2d(2)
print(pool2d(X))