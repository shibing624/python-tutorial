# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
import time

import matplotlib.pyplot as plt
import numpy as np

# plt.ion() #开启interactive mode
x = np.linspace(0, 50, 1000)
plt.figure(1)  # 创建图表1
plt.plot(x, np.sin(x))
plt.show()
time.sleep(2)
plt.close(1)
plt.figure(2)  # 创建图表2
plt.plot(x, np.cos(x))
plt.draw()
time.sleep(1)
print('it is ok')
