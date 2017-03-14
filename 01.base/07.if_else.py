# -*- coding: utf-8 -*-
"""
@description: 判断语句
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 基本用法
a = 62
print("exam score check:")
if a >= 60:
    print("student pass")
elif a == 0:
    print("student 0: not pass")
else:
    print("student not pass")

# 可以使用 and ， or , not 等关键词结合多个判断条件：
a = 10
b = -5
print(a > 0 and b < 0)  # True
print(not a > 0)  # False
print(a < 0 or b < 0)  # True

# 一个例子
year = 1900
if year % 400 == 0:
    print("This is a leap year!")
# 两个条件都满足才执行
elif year % 4 == 0 and year % 100 != 0:
    print("This is a leap year!")
else:
    print("This is not a leap year.")
# This is not a leap year.

my_list = [1, 2]
# 判断一个列表是否为空。
if len(my_list) > 0:
    print("the first element is: ", my_list[0])
else:
    print("no element.")
