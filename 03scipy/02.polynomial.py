# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("C-067.txt",
                     delimiter="\t",  # TAB 分隔
                     skip_header=2)

print(data[:7])
print(data[0])
p = plt.plot(data[0], data[1], 'kx')
t = plt.title("janaf data for mechane")
a = plt.axis([0, 6000, 30, 120])
x = plt.xlabel("temperature (K)")
y = plt.ylabel(r"$C_p$ ($\frac{kJ}{kg K}$)")
# plt.show()
plt.close()

# 径向基函数
x = np.linspace(-3, 3, 100)
# 高斯函数
plt.plot(x, np.exp(-1 * x ** 2))
t = plt.title("Gaussian")
plt.savefig('Gaussian.png')
plt.show()

# 高维 RBF 插值
# 三维数据点：
x, y = np.mgrid[-np.pi / 2:np.pi / 2:5j, -np.pi / 2:np.pi / 2:5j]
z = np.cos(np.sqrt(x ** 2 + y ** 2))
fig = plt.figure(figsize=(12, 6))
ax = fig.gca(projection="3d")
ax.scatter(x, y, z)
fig.savefig("mplot3d.jpg")
plt.show()
