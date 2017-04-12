# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 子模块	描述
# cluster	聚类算法
# constants	物理数学常数
# fftpack	快速傅里叶变换
# integrate	积分和常微分方程求解
# interpolate	插值
# io	输入输出
# linalg	线性代数
# odr	正交距离回归
# optimize	优化和求根
# signal	信号处理
# sparse	稀疏矩阵
# spatial	空间数据结构和算法
# special	特殊方程
# stats	统计分布和函数
# weave	C/C++ 积分
# 使用scipy之前，基础模块需要导入：
import numpy as np
# 使用scipy的子模块时，需要导入：
from scipy import optimize

print(np.info(optimize))
