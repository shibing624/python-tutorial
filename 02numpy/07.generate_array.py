# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# 生成数组的函数
# arange 生成数组，[start,stop)
# arange(start, stop=None, step=1, dtype=None)
print(np.arange(5))  # [0 1 2 3 4]

a = np.arange(0, 2 * np.pi, np.pi / 4)
print(a)

# linspace
# linspace(start,stop,N)
# 产生N个等距分布在[start,stop]间的元素组成的数组，包括start,stop
print(np.linspace(0, 1, 5))  # [ 0.    0.25  0.5   0.75  1.  ]

# logspace
# logspace(start, stop, N)
# 产生 N 个对数等距分布的数组，默认以10为底：
print(np.logspace(0, 1, 5))
# 产生的值为$\left[10^0, 10^{0.25},10^{0.5},10^{0.75},10^1\right]$。

# 二维平面中生成一个网格:meshgrid
x_ticks = np.linspace(-1, 1, 5)
y_ticks = np.linspace(-1, 1, 5)
x, y = np.meshgrid(x_ticks, y_ticks)
print(x_ticks)
print(x)

# 图例
import matplotlib.pyplot as plt
from matplotlib import cm


def f(x, y):
    # sinc 函数
    r = np.sqrt(x ** 2 + y ** 2)
    result = np.sin(r) / r
    result[r == 0] = 1.0
    return result


x_ticks = np.linspace(-10, 10, 51)
y_ticks = np.linspace(-10, 10, 51)

x, y = np.meshgrid(x_ticks, y_ticks, sparse=True)
print(x)  # x, y 中有很多冗余的元素，这里提供了一个 sparse 的选项去冗余
z = f(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,
                rstride=1, cstride=1,
                cmap=cm.YlGnBu_r)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
