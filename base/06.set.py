# -*- coding: utf-8 -*-
"""
@description: set 方法操作
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

# 集合操作
a = {1, 2, 3, 4}
b = {2, 3, 4, 5}

# 并
# 两个集合的并，返回包含两个集合所有元素的集合（去除重复）。
# 可以用方法 a.union(b) 或者操作 a | b 实现。
c = a.union(b)
print(c)  # {1, 2, 3, 4, 5, 6}

# 操作 a | b 实现
d = a | b
print(c)

print(c == d)

# 交
# 两个集合的交，返回包含两个集合共有元素的集合。
# 可以用方法 a.intersection(b) 或者操作 a & b 实现。
c = a.intersection(b)
print(c)  # set([2, 3, 4])

d = a & b
print(d)

# 差
# a 和 b 的差集，返回只在 a 不在 b 的元素组成的集合。
# 可以用方法 a.difference(b) 或者操作 a - b 实现。
c = a.difference(b)
print(c)  # set([1])
d = a - b
print(d)

# 对称差
# a 和b 的对称差集，返回在 a 或在 b 中，但是不同时在 a 和 b 中的元素组成的集合。
# 可以用方法 a.symmetric_difference(b) 或者操作 a ^ b 实现（异或操作符）。
c = a.symmetric_difference(b)
print(c)  # set([1, 5])

d = a ^ b
print(d)

# 包含关系
a = {1, 2, 3}
b = {1, 2}
# 要判断 b 是不是 a 的子集，可以用 b.issubset(a) 方法，
# 或者更简单的用操作 b <= a ：
c = b.issubset(a)
print(c)  # True
d = (b <= a)
print(d)

# 也可以用 a.issuperset(b) 或者 a >= b 来判断：
print(a >= b)

# 方法只能用来测试子集，但是操作符可以用来判断真子集：
print(a < a)  # False
print(a <= a)  # True

# 集合方法
# add 方法向集合添加单个元素
# 跟列表的 append 方法类似，用来向集合添加单个元素。
# s.add(a) 将元素 a 加入集合 s 中。
s = {1, 3, 4}
s.add(4)
print(s)  # set([1, 3, 4])

s.add(5)
print(s)  # set([1, 3, 4, 5])

# update 方法向集合添加多个元素
# 跟列表的extend方法类似，用来向集合添加多个元素。
# s.update(seq)
s.update([10, 11, 12])
print(s)  # set([1, 3, 4, 5, 10, 11, 12])

# remove 方法移除单个元素
s = {1, 3, 4}
s.remove(1)
print(s)  # set([3, 4])

# pop 方法弹出元素
# 由于集合没有顺序，不能像列表一样按照位置弹出元素，
# 所以 pop 方法删除并返回集合中任意一个元素，如果集合中没有元素会报错。
s = {1, 3, 4}
d = s.pop()
print(s, d)

# discard 方法作用与 remove 一样
s = {1, 3, 4}
s.discard(3)
print(s)  # set([1, 4])

# difference_update方法
# a.difference_update(b) 从a中去除所有属于b的元素：
a = {1, 2, 3, 4}
b = {2, 3, 4, 5}
a.difference_update(b)
print(a)  # set([1])
