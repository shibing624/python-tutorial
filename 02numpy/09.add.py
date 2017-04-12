# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.arange(2)
print(a, b)

a_list = list(a)
b_list = list(b)
d = a_list + b_list
print(d)
c = a_list.extend(b_list)
print(c)
print(a_list)
print(np.array(a_list))

print(np.append(a, b))

k = []
k.append(b_list)
print(k)

k_list = [1, 2]
k.extend(k_list)
print(k)
