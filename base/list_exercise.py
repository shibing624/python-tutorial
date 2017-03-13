# -*- coding: utf-8 -*-
"""
@description: 介绍列表的方法及示例演示其使用，包括：长度、修改列表、取值、排序
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

empty_list = list()
print(empty_list)

# 用 len 查看列表长度：
a = [1, 2, 3]
b = [2, 3, 'hello']
c = a + b
print(c)  # [1, 2, 3, 2, 3, u'hello']

d = b * 2
print(d)  # [2, 3, u'hello', 2, 3, u'hello']

print(d[-1])

print(a)
# 修改列表
a[0] = 100
print(a)

# 这种赋值也适用于分片，例如，将列表的第2，3两个元素换掉：
a[1:3] = [200, 300]
print(a)

# 事实上，对于连续的分片（即步长为 1 ），Python采用的是整段替换的方法，
# 两者的元素个数并不需要相同，
# 例如，将 [11,12] 替换为 [1,2,3,4]：
a = [10, 11, 12, 13, 14]
a[1:3] = [1, 2, 3, 4]
print(a)  # [10, 1, 2, 3, 4, 13, 14]

# 用这种方法来删除列表中一个连续的分片：
a = [10, 1, 2, 11, 12]
print(a[1:3])
a[1:3] = []
print(a)

# 对于不连续（间隔step不为1）的片段进行修改时，两者的元素数目必须一致：
a = [10, 11, 12, 13, 14]
a[::2] = [1, 2, 3]
print(a)  # [1, 11, 2, 13, 3]

# Python提供了删除列表中元素的方法 'del'。
a = [100, 'a', 'b', 200]
del a[0]
print(a)  # [u'a', u'b', 200]

# 删除间隔的元素：
a = ['a', 1, 'b', 2, 'c']
del a[::2]
print(a)  # [1, 2]

# 用 in 来看某个元素是否在某个序列（不仅仅是列表）中，
# 用not in来判断是否不在某个序列中。
a = [1, 2, 3, 4, 5]
print(1 in a)
print(1 not in a)

# 也可以作用于字符串：
s = 'hello world'
print("'he' in s : ", 'he' in s)  # True
print("'world' not in s : ", 'world' not in s)  # False

# 列表中可以包含各种对象，甚至可以包含列表：
a = [1, 2, 'six', [3, 4]]
print(a[3])  # [3,4]
# a[3]是列表，可以对它再进行索引：
print(a[3][1])  # 4

# 列表方法

# 列表中某个元素个数
a = [1, 1, 2, 3, 4, 5]
print(len(a))  # 总个数：6
# 元素1出现的个数
print(a.count(1))  # 2
# l.index(ob) 返回列表中元素 ob 第一次出现的索引位置，如果 ob 不在 l 中会报错。
print(a.index(1))  # 0

# 向列表添加单个元素
# l.append(ob) 将元素 ob 添加到列表 l 的最后。
a = [1, 1, 2, 3, 4, 5]
a.append(10)
print(a)  # [1, 1, 2, 3, 4, 5, 10]

# append每次只添加一个元素，并不会因为这个元素是序列而将其展开：
a.append([11, 12])
print(a)  # [1, 1, 2, 3, 4, 5, 10, [11, 12]]

# 向列表添加序列
# l.extend(lst) 将序列 lst 的元素依次添加到列表 l 的最后，作用相当于 l += lst。
a = [1, 2, 3, 4]
a.extend([6, 7, 1])
print(a)  # [1, 2, 3, 4, 6, 7, 1]

# 插入元素
# l.insert(idx, ob) 在索引 idx 处插入 ob ，之后的元素依次后移。
a = [1, 2, 3, 4]
# 在索引 3 插入 'a'
a.insert(3, 'a')
print(a)  # [1, 2, 3, u'a', 4]

# 移除元素
# l.remove(ob) 会将列表中第一个出现的 ob 删除，如果 ob 不在 l 中会报错。
a = [1, 1, 2, 3, 4]
# 移除第一个1
a.remove(1)
print(a)  # [1, 2, 3, 4]

# 弹出元素
# l.pop(idx) 会将索引 idx 处的元素删除，并返回这个元素。
a = [1, 2, 3, 4]
b = a.pop(0)  # 1
print('pop:', b, ' ;result:', a)

# 排序
# l.sort() 会将列表中的元素按照一定的规则排序：
a = [10, 1, 11, 13, 11, 2]
a.sort()
print(a)  # [1, 2, 10, 11, 11, 13]

# 如果不想改变原来列表中的值，可以使用 sorted 函数：
a = [10, 1, 11, 13, 11, 2]
b = sorted(a)
print(a)  # [10, 1, 11, 13, 11, 2]
print(b)  # [1, 2, 10, 11, 11, 13]

# 列表反向
# l.reverse() 会将列表中的元素从后向前排列。
a = [1, 2, 3, 4, 5, 6]
a.reverse()
print(a)  # [6, 5, 4, 3, 2, 1]

# 如果不想改变原来列表中的值，可以使用这样的方法：
a = [1, 2, 3, 4, 5, 6]
b = a[::-1]
print(a)
print(b)
# 如果不清楚用法，可以查看帮助：a.sort?
