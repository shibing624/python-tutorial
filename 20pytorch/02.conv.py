# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import torch
from torch import nn


def corr2d(X, K):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
Y = torch.tensor([[0, 1], [2, 3]])
Z = corr2d(X, Y)
print(Z)

print(torch.randn(1))


class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.b


X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -2]])
Y = corr2d(X, K)
print(Y)

# 5.1.4 通过数据学习核数组
# 最后我们来看一个例子，它使用物体边缘检测中的输入数据X和输出数据Y来学习我们构造的核数组K。
# 我们首先构造一个卷积层，其卷积核将被初始化成随机数组。接下来在每一次迭代中，我们使用平方误差来比较Y和卷积层的输出，
# 然后计算梯度来更新权重。


conv2d = Conv2d(kernel_size=(1, 2))
step = 40
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.b.data -= lr * conv2d.b.grad

    # 清理梯度
    conv2d.weight.grad.fill_(0)
    conv2d.b.grad.fill_(0)

    if (i + 1) % 5 == 0:
        print(f'Step {i + 1}, loss {l.item():.4f}')

# 可以看到，40次迭代后误差已经降到了一个比较小的值。现在来看一下学习到的卷积核的参数。
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.b.data)