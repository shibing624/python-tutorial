# -*- coding: utf-8 -*-
"""
@description: ；列表推导式
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import time

# 循环可以用来生成列表：
values = [2, 2, 3]
squares = []
for x in values:
    squares.append(x ** 2)
print(squares)  # [4, 4, 9]

# 列表推导式可以使用更简单的方法来创建这个列表：
values = [3, 8, 10, 14]
squares = [x ** 2 for x in values]
print(squares)  # [9, 64, 100, 196]

# 可以加入条件筛选，在上面的例子中，假如只想保留列表中不大于8的数的平方：
squares = [x ** 2 for x in values if x <= 10]
print(squares)  # [9, 64, 100]

# 平方的结果不大于100的：
squares = [x ** 2 for x in values if x ** 2 <= 80]
print(squares)  # [9, 64]

# 也可以使用推导式生成集合和字典：
values = [10, 21, 4, 7, 12]
square_set = {x ** 2 for x in values if x <= 10}
print(square_set)  # set([16, 49, 100])
square_dict = {x: x ** 2 for x in values if x <= 10}
print(square_dict)  # {10: 100, 4: 16, 7: 49}

# 计算上面例子中生成的列表中所有元素的和：
total = sum([x ** 2 for x in values if x < 10])
print(total)  # 65

# 但是，Python会生成这个列表，然后在将它放到垃圾回收机制中（因为没有变量指向它），
# 这毫无疑问是种浪费。
# 为了解决这种问题，与xrange()类似，Python使用产生式表达式来解决这个问题：
total = sum(x ** 2 for x in values if x < 10)
print(total)  # 65
# 与上面相比，只是去掉了括号，但这里并不会一次性的生成这个列表。

# 比较一下两者的用时：
x = range(1000000)
t1 = time.time()
total = sum([x ** 2 for x in values if x < 10])
print("list speed: ", time.time() - t1)

t2 = time.time()
total = sum(x ** 2 for x in values if x < 10)
print("comprehension speed:", time.time() - t2)

# ipython 下可以输入:
# x = range(1000000)
# %timeit total = sum([i**2 for i in x])
# %timeit total = sum(i**2 for i in x)
