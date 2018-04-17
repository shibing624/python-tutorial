# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#variable
import torch
from torch.autograd import Variable

a = Variable(torch.ones(3, 3), requires_grad=True)
print(a)

b = a + 2
print(b)

c = b * b * 2
out = c.mean()
print(c, out)

print('a.grad:')
print(a.grad)

print('a.grad after backward:')
out.backward()
print(a.grad)

print('y = y * 2 result:')
x = torch.randn(3)
print('raw x:',x)
x = Variable(x, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
