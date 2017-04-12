# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 模块
# Python会将所有 .py 结尾的文件认定为Python代码文件

# __name__ 属性
# 有时候我们想将一个 .py 文件既当作脚本，又能当作模块用，
# 这个时候可以使用 __name__ 这个属性。
PI = 3.14


def get_sum(lst):
    """
    Sum the values in the list
    :param lst:
    :return:
    """
    total = 0
    for v in lst:
        total = total + v
    return total


def test():
    l = [1, 2, 3]
    assert (get_sum(l) == 6)
    print("test pass.")


if __name__ == '__main__':
    test()

# 上文保存为ex.py
# 其他导入方法
# 可以从模块中导入变ex量：
from ex import PI, get_sum

print(PI)  # 3.14
print(get_sum([2, 3]))  # 5

# 或者使用 * 导入所有变量,不提倡，可能覆盖一些已有的函数

# 删除文件：
import os

os.remove('ex2.py')

# 包
# 导入包要求：

# 文件夹 foo 在Python的搜索路径中
# __init__.py 表示 foo 是一个包，它可以是个空文件。
