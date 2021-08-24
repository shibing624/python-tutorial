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
print('raw x:', x)
x = Variable(x, requires_grad=True)
y = x * 2
while y.data.norm() < 10:
    y = y * 2
print(y)

print('*' * 43)
a = torch.randn(1, 3)
print('a:', a, a.shape)
b = torch.rand(1, 3)
print('b:', b, b.shape)

aa = torch.unsqueeze(a, 1)
print('aa:', aa, aa.shape)
bb = torch.unsqueeze(a, 0)
print('bb:', bb, bb.shape)
print('a:', a, a.shape)
dd = a.unsqueeze(0)
print('a unsqueeze:', a, a.shape)
print('dd:', dd, dd.shape)

c = dd.squeeze(0)
print('c:', c, dd, c.shape, dd.shape)
d = torch.squeeze(dd, 0)
print('d:', d, d.shape)
e = dd.squeeze()
print('dd:', dd.shape, dd, e, e.shape)
f = dd.squeeze(0)
print('f:', f, f.shape)
