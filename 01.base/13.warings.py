# -*- coding: utf-8 -*-
"""
@description:警告
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 引入警告模块
import warnings


# 我们使用 warnings 中的 warn 函数：
# warn(msg, WarningType = UserWarning)
def month_warining(m):
    if not 1 <= m <= 12:
        msg = "month (%d) is not between 1 and 12 " % m
        warnings.warn(msg, RuntimeWarning)


month_warining(13)  # 报警告

# 有时候我们想要忽略特定类型的警告，可以使用 warnings 的 filterwarnings 函数：
# filterwarnings(action, category)
# 将 action 设置为 'ignore' 便可以忽略特定类型的警告：
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
month_warining(13)  # 不报警告
