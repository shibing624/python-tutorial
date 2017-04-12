# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 列表是可变的（Mutable）
a = [1, 2, 3, 4]
a[0] = 100
a.insert(3, 200)
print(a)  # [100, 2, 3, 200, 4]

# 字符串是不可变的（Immutable）
s = "hello world"
# 通过索引改变会报错

# 字符串方法只是返回一个新字符串，并不改变原来的值：
print(s.replace('world', 'Mars'))  # hello Mars
print(s)  # hello world

# 如果想改变字符串的值，可以用重新赋值的方法：
s = s.replace('world', 'YunYun')
print(s)  # hello YunYun

# 可变数据类型: list, dictionary, set, numpy array, user defined objects
# 不可变数据类型: integer, float, long, complex, string, tuple, frozenset