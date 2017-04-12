# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

__author__ = 'XuMing'
# Python 中常用的统计工具有 Numpy, Pandas, PyMC, StatsModels 等。
# Scipy 中的子库 scipy.stats 中包含很多统计上的方法。
from numpy import *
from matplotlib import pyplot

# Numpy 自带简单的统计方法：
heights = array([1.46, 1.79, 2.01, 1.75, 1.56, 1.69, 1.88, 1.76, 1.88, 1.78])
print('mean,', heights.mean())
print('min,', heights.min())
print('max', heights.max())
print('stand deviation,', heights.std())

# 导入 Scipy 的统计模块：
import scipy.stats.stats as st

print('median, ', st.nanmedian(heights))  # 忽略nan值之后的中位数
print('mode, ', st.mode(heights))  # 众数及其出现次数
print('skewness, ', st.skew(heights))  # 偏度
print('kurtosis, ', st.kurtosis(heights))  # 峰度

# 概率分布
# 常见的连续概率分布有：
# 均匀分布
# 正态分布
# 学生t分布
# F分布
# Gamma分布
# ...

# 离散概率分布：
# 伯努利分布
# 几何分布
# ...
# 这些都可以在 scipy.stats 中找到。

# 正态分布
from scipy.stats import norm

# 它包含四类常用的函数：
#
# norm.cdf 返回对应的累计分布函数值
# norm.pdf 返回对应的概率密度函数值
# norm.rvs 产生指定参数的随机变量
# norm.fit 返回给定数据下，各参数的最大似然估计（MLE）值

# 从正态分布产生500个随机点：
x_norm = norm.rvs(size=500)
type(x_norm)
# pyplot.ion() #开启interactive mode
# 直方图：
h = pyplot.hist(x_norm)
print('counts, ', h[0])
print('bin centers', h[1])
figure = pyplot.figure(1)  # 创建图表1
pyplot.show()

# 归一化直方图（用出现频率代替次数），将划分区间变为 20（默认 10）：
h = pyplot.hist(x_norm, normed=True, bins=20)
pyplot.show()
# 在这组数据下，正态分布参数的最大似然估计值为：
x_mean, x_std = norm.fit(x_norm)

print('mean, ', x_mean)
print('x_std, ', x_std)

# 将真实的概率密度函数与直方图进行比较：
h = pyplot.hist(x_norm, normed=True, bins=20)

x = linspace(-3, 3, 50)
p = pyplot.plot(x, norm.pdf(x), 'r-')
pyplot.show()

# 导入积分函数：
from scipy.integrate import trapz

x1 = linspace(-2, 2, 108)
p = trapz(norm.pdf(x1), x1)
print('{:.2%} of the values lie between -2 and 2'.format(p))

pyplot.fill_between(x1, norm.pdf(x1), color='red')
pyplot.plot(x, norm.pdf(x), 'k-')
pyplot.show()

# 可以通过 loc 和 scale 来调整这些参数，一种方法是调用相关函数时进行输入：
x = linspace(-3, 3, 50)
p = pyplot.plot(x, norm.pdf(x, loc=0, scale=1))
p = pyplot.plot(x, norm.pdf(x, loc=0.5, scale=2))
p = pyplot.plot(x, norm.pdf(x, loc=-0.5, scale=.5))
pyplot.show()

# 不同参数的对数正态分布：
from scipy.stats import lognorm

x = linspace(0.01, 3, 100)

pyplot.plot(x, lognorm.pdf(x, 1), label='s=1')
pyplot.plot(x, lognorm.pdf(x, 2), label='s=2')
pyplot.plot(x, lognorm.pdf(x, .1), label='s=0.1')

pyplot.legend()
pyplot.show()

# 离散分布
from scipy.stats import randint

# 离散均匀分布的概率质量函数（PMF）：
high = 10
low = -10

x = arange(low, high + 1, 0.5)
p = pyplot.stem(x, randint(low, high).pmf(x))  # 杆状图
pyplot.show()

# 假设检验
# 导入相关的函数：
#
# 1.正态分布
# 2.独立双样本 t 检验，配对样本 t 检验，单样本 t 检验
# 3.学生 t 分布

from scipy.stats import norm
from scipy.stats import ttest_ind

# 独立样本 t 检验
# 两组参数不同的正态分布：
n1 = norm(loc=0.3, scale=1.0)
n2 = norm(loc=0, scale=1.0)
# 从分布中产生两组随机样本：
n1_samples = n1.rvs(size=100)
n2_samples = n2.rvs(size=100)
# 将两组样本混合在一起：
samples = hstack((n1_samples, n2_samples))
# 最大似然参数估计：
loc, scale = norm.fit(samples)
n = norm(loc=loc, scale=scale)
# 比较：
x = linspace(-3, 3, 100)

pyplot.hist([samples, n1_samples, n2_samples], normed=True)
pyplot.plot(x, n.pdf(x), 'b-')
pyplot.plot(x, n1.pdf(x), 'g-')
pyplot.plot(x, n2.pdf(x), 'r-')
pyplot.show()

# 独立双样本 t 检验的目的在于判断两组样本之间是否有显著差异：
t_val, p = ttest_ind(n1_samples, n2_samples)

print('t = {}'.format(t_val))
print('p-value = {}'.format(p))
# t = 0.868384594123
# p-value = 0.386235148899
# p 值小，说明这两个样本有显著性差异。
