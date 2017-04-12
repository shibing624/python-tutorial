# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
from numpy import *

# plot 二维图
# plot(y)
# plot(x, y)
# plot(x, y, format_string)

x = linspace(0, 2 * pi, 50)
print(x)
plt.plot(sin(x))
plt.show()

# 给定 x 和 y 值：
plt.plot(x, sin(x))
plt.show()

# 多条数据线：
plt.plot(x, sin(x), x, sin(2 * x))
plt.show()

# 使用字符串，给定线条参数：
plt.plot(x, sin(x), 'r-^')
plt.show()

# 多线条：
plt.plot(x, sin(x), 'b-o', x, sin(2 * x), 'r-^')
plt.show()

# scatter 散点图
# scatter(x, y)
# scatter(x, y, size)
# scatter(x, y, size, color)
plt.plot(x, sin(x), 'bo')
plt.show()  # 二维散点图

# 用scatter画图
plt.scatter(x, sin(x))
plt.show()  # 正弦函数

# 事实上，scatter函数与Matlab的用法相同，还可以指定它的大小，颜色等参数：

# 标签
# 可以在 plot 中加入 label ，使用 legend 加上图例：

t = linspace(0, 2 * pi, 50)
x = sin(t)
plt.plot(t, x, 'bo', t, sin(2 * t), 'r-^', label='sin', color='red', )
plt.legend()
plt.xlabel('radians')
plt.ylabel('amplitude', fontsize='large')
plt.title('Sin(x)')
plt.grid()
plt.show()

# 直方图
data = array([1234, 321, 400, 120, 11, 30, 2000])
plt.hist(data, 7)
plt.show()
