# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 列表和字符串都是一种有序序列，而集合 set 是一种无序的序列。
# 因为集合是无序的，所以当集合中存在两个同样的元素的时候，只会保存其中的一个（唯一性）；
# 同时为了确保其中不包含同样的元素，集合中放入的元素只能是不可变的对象（确定性）。

# 可以用set()函数来显示的生成空集合：
a = set()
print(type(a))

# 使用一个列表来初始化一个集合：
a = set([1, 2, 3, 1])
print(a)  # 集合会自动去除重复元素 1。

# 集合中的元素是用大括号{}包含起来的，这意味着可以用{}的形式来创建集合：
a = {1, 2, 3, 1}
print(a)  # {1, 2, 3}

# 但是创建空集合的时候只能用set来创建，因为在Python中{}创建的是一个空的字典：
s = {}
print(type(s))  # <type 'dict'>
