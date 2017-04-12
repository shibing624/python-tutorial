# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

from numpy import *

# 查看形状，会返回一个元组，每个元素代表这一维的元素数目：
a = array([1, 2, 3, 5])
# 1维数组，返回一个元组
print(a.shape)

# 查看元素数目：
print(a.size)

# 使用fill方法设定初始值
# 可以使用 fill 方法将数组设为指定值：
print(a)
a.fill(-4)
print(a)

# 切片，支持负索引：
a = array([11, 12, 13, 14, 15])
print(a[1:-2])  # [12 13]

# 省略参数：
print(a[::2])  # [11 13 15]
print(a[-2:])  # array([14, 15])

# 假设我们记录一辆汽车表盘上每天显示的里程数：
rec = array([21000, 21180, 21240, 22100, 22400])
dist = rec[1:] - rec[:-1]
print(dist)

# 多维数组
a = array([[1, 2, 3], [7, 8, 9]])
print(a)
# 查看形状：
print(a.shape)
# 查看总的元素个数：
print(a.size)
# 查看维数：
print(a.ndim)
# 对于二维数组，可以传入两个数字来索引：
print(a[1, 1])
# 索引一整行内容：
print(a[0])

# 多维数组
a = array([[0, 1, 2, 3, 4, 5],
           [10, 11, 12, 13, 14, 15],
           [20, 21, 22, 23, 24, 25],
           [30, 31, 32, 33, 34, 35],
           [40, 41, 42, 43, 44, 45],
           [50, 51, 52, 53, 54, 55]])
# 想得到第一行的第 4 和第 5 两个元素：
print(a[0, 3:5])  # [3 4]
# 得到最后两行的最后两列：
print(a[4:, 4:])  # [[44 45][54 55]]
# 得到第三列：
print(a[:, 2])  # [ 2 12 22 32 42 52]
# 取出3，5行的奇数列：
b = a[2::2, ::2]
print(b)

# 切片在内存中使用的是引用机制。
# 引用机制意味着，Python并没有为 b 分配新的空间来存储它的值，
# 而是让 b 指向了 a 所分配的内存空间，因此，改变 b 会改变 a 的值：
a = array([0, 1, 2, 3, 4])
b = a[2:4]
print(b)

b[0] = 10
print(a)

# 而这种现象在列表中并不会出现：
b = a[2:3]
b[0] = 12
print(a)
# 解决方法是使用copy()方法产生一个复制，这个复制会申请新的内存：
b = a[2:4].copy()
b[0] = 10
print(a, b)

# 一维花式索引
# 与 range 函数类似，我们可以使用 arange 函数来产生等差数组。
a = arange(0, 80, 10)
print(a)

# 花式索引需要指定索引位置：
indices = [1, 2, -3]
y = a[indices]
print(y)

# 还可以使用布尔数组来花式索引：
mask = array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
print(a[mask])  # [10 20 50 70]

# 选出了所有大于0.5的值：
from numpy.random import rand

a = rand(10)
print(a)

mask = a > 0.5
print(a[mask])

# “不完全”索引
# 只给定行索引的时候，返回整行：
a = array([[0, 1, 2, 3, 4, 5],
           [10, 11, 12, 13, 14, 15],
           [20, 21, 22, 23, 24, 25],
           [30, 31, 32, 33, 34, 35],
           [40, 41, 42, 43, 44, 45],
           [50, 51, 52, 53, 54, 55]])
b = a[:3]
print(b)

# 这时候也可以使用花式索引取出第2，3，5行：
condition = array([0, 1, 1, 0, 1, 0], dtype=bool)
c = a[condition]
print(c)

# where语句
# where(array)
# where 函数会返回所有非零元素的索引。
a = array([1, 2, 4, 6])
print(a > 2)  # [False False  True  True]

b = where(a > 2)
print(b)
# 注意到 where 的返回值是一个元组。
index = where(a > 2)[0]
print(index)  # [2 3]
# 可以直接用 where 的返回值进行索引：
loc = where(a > 2)
b = a[loc]
print(b)  # [4 6]

# 考虑二维数组：
a = array([[0, 12, 5, 20],
           [1, 2, 11, 15]])
loc = where(a > 10)
print(loc)  # (array([0, 0, 1, 1]), array([1, 3, 2, 3]))
# 也可以直接用来索引a：
b = a[loc]
print(b)  # [12 20 11 15]
# 或者可以这样：
rows, cols = where(a > 10)
print(rows)
print(cols)
print(a[rows, cols])

# 例子：
a = arange(20)
a.shape = 5, 4
print(a)
print(a > 12)
b = where(a > 12)
print(b)
# (array([3, 3, 3, 4, 4, 4, 4]), array([1, 2, 3, 0, 1, 2, 3]))

print(a[b])  # [13 14 15 16 17 18 19]
