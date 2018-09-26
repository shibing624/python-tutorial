# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals
import tracemalloc

tracemalloc.start(10)

time1 = tracemalloc.take_snapshot()
# 导入numpy
# 很多其他科学计算的第三方库都是以Numpy为基础建立的。
# Numpy的一个重要特性是它的数组计算。
from numpy import *


# 使用前一定要先导入 Numpy 包，导入的方法有以下几种：
# import numpy
# import numpy as np
# from numpy import *
# from numpy import array, sin

# 假如我们想将列表中的每个元素增加1，但列表不支持这样的操作（报错）：
a = [1, 2]
# print(a + 1) # 报错

# 使用numpy.array
a = array(a)
print(a)  # [1 2]

b = a + 1
print(b)  # array([2,3])

# 与另一个 array 相加，得到对应元素相加的结果：
c = a + b
print(c)  # array([3,5])

# 对应元素相乘：
print(a * b)  # [2 6]

# 对应元素乘方：
print(a ** b)  # [1 8]

# 提取数组中的元素
# 提取第一个
a = array([1, 2, 3, 4])
print(a[0])  # 1

# 提取前两个元素：
print(a[:2])  # [1 2]

# 最后两个元素
print(a[-2:])  # [3 4]

# 相加：
print(a[:2] + a[-2:])  # [4 6]

# 修改数组形状
# 查看array的形状：
b = a.shape
print(b)  # (4,)

# 修改 array 的形状：
a.shape = 2, 2
print(a)
# [[1 2]
# [3 4]]

# 多维数组
# a 现在变成了一个二维的数组，可以进行加法：
print(a + a)
# [[2 4]
#  [6 8]]

# 乘法仍然是对应元素的乘积，并不是按照矩阵乘法来计算：
print(a * a)
# [[ 1  4]
# [ 9 16]]

# 画图
# linspace 用来生成一组等间隔的数据：
a = linspace(0, 2 * pi, 10)
print(a)
# [ 0.          0.6981317   1.3962634   2.0943951   2.7925268   3.4906585
#   4.1887902   4.88692191  5.58505361  6.28318531]

# 三角函数
b = sin(a)
print(b)


# 画图
from matplotlib import pyplot as plt

plt.plot(a, b)
# plt.show()

# 从数组中选择元素
# 假设我们想选取数组b中所有非负的部分，首先可以利用 b 产生一组布尔值：
mask = b >= 0
print(mask)


time2 = tracemalloc.take_snapshot()

stats = time2.compare_to(time1, 'lineno')
print('*'*32)
for stat in stats[:3]:
    print(stat)

stats = time2.compare_to(time1, 'traceback')
print('*'*32)
for stat in stats[:3]:
    print(stat.traceback.format())