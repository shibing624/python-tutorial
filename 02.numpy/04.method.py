# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

from numpy import *

a = array([[0, 1, 2, 3], [4, 5, 6, 7]])
print(a)

for row in a:
    print(row)

# 所有元素的迭代器：
for i in a.flat:
    print(i)


# 转置：
print(a.T)
print(a)
print(a.shape)  # 数组形状 (m,n,o,...)

print(a.size)  # 数组元素数
a.resize((4, 2))
print(a)
print(a.shape)

# 去除长度为1的维度：
a = a.squeeze()
print(a.shape)

# 复制：
a = array([[0, 1, 2, 3], [4, 5, 6, 7]])
b = a.copy()
b[0][0] = -1
print(b)

# 填充：
b.fill(9)
print(b)

# 转化为列表：
print(a.tolist())

# 复数
# 实部：
b = array([1 + 2j, 3 + 4j, 5 + 6j])
c = b.real
print(c)

# 虚部：
d = b.imag
print(d)

# 共轭：
print(b.conj())

# 保存成文本：
a.dump("file.txt")

# 字符串：
a.dumps()

# 写入文件
a.tofile('foo.csv', sep=',', format="%s")

# 查找排序
# 非零元素的索引：
b = a.nonzero()
print(a, b)

# 排序：
b = array([3, 2, 7, 4, 1])
b.sort()
print(b)

# 排序的索引位置：
b = array([2, 3, 1])
print(b.argsort(axis=-1))  # array([2, 0, 1], dtype=int64)
# 将 b 插入 a 中的索引，使得 a 保持有序：
a = array([1, 3, 4, 6])
b = array([0, 2, 5])
print(a.searchsorted(b))

# 元素数学操作
# 限制在一定范围：
a = array([[4, 1, 3], [2, 1, 5]])
print(a.clip(0, 2))

# 近似：
a = array([1.344, 2.449, 2.558])
b = a.round(decimals=2)
print(b)  # [ 1.34  2.45  2.56]

# 是否全部非零：
print(a.all())

import os

os.remove('foo.csv')
os.remove('file.txt')
