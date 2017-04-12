# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import pprint

# 字典 dictionary ，在一些编程语言中也称为 hash ， map ，
# 是一种由键值对组成的数据结构。

a = {}
print(type(a))  # <type 'dict'>
a = dict()
print(type(a))

# 插入键值
a['f'] = 'num 1'
a['s'] = 'num 2'
print(a)  # {u's': u'num 2', u'f': u'num 1'}

# 查看键值
print(a['s'])  # num 2

# 更新
a['f'] = 'num 3'
print(a)  # {u's': u'num 2', u'f': u'num 3'}

# 初始化字典
a = {'first': 'num 1', 'second': 'num 2', 3: 'num 3'}
print(a['first'])  # num 1
print(a[3])  # num 3

# Python中不能用支持用数字索引按顺序查看字典中的值，
# 而且数字本身也有可能成为键值，这样会引起混淆:
# a[0] 会报错

# 例子
# 定义四个字典
e1 = {'mag': 0.05, 'width': 20}
e2 = {'mag': 0.04, 'width': 25}
e3 = {'mag': 0.05, 'width': 80}
e4 = {'mag': 0.03, 'width': 30}
# 以字典作为值传入新的字典
events = {500: e1, 760: e2, 3001: e3, 4180: e4}
# {760: {u'width': 25, u'mag': 0.04},
# 3001: {u'width': 80, u'mag': 0.05},
# 500: {u'width': 20, u'mag': 0.05},
# 4180: {u'width': 30, u'mag': 0.03}}
print(events)

# 另一个例子
people = [
    {'first': 'Sam', 'last': 'Malone', 'name': 35},
    {'first': 'Woody', 'last': 'Boyd', 'name': 21},
    {'first': 'Norm', 'last': 'Peterson', 'name': 34},
    {'first': 'Diane', 'last': 'Chambers', 'name': 33}
]
# [{'first': 'Sam', 'last': 'Malone', 'name': 35},
#  {'first': 'Woody', 'last': 'Boyd', 'name': 21},
#  {'first': 'Norm', 'last': 'Peterson', 'name': 34},
#  {'first': 'Diane', 'last': 'Chambers', 'name': 33}]
print(people)

# 使用 dict 初始化字典
# 除了通常的定义方式，还可以通过 dict() 转化来生成字典：
my_dict = dict([('name', 'lili'),
                ('sex', 'female'),
                ('age', 32),
                ('address', 'beijing')])
# {u'age': 32,
# u'address': u'beijing',
# u'name': u'lili',
# u'sex': u'female'}
print(my_dict)

# 利用索引直接更新键值对：
my_dict['age'] += 1
print(my_dict)  # u'age': 33

# 可以使用元组作为键值，
# 例如，可以用元组做键来表示从第一个城市飞往第二个城市航班数的多少：
connections = {}
connections[('New York', 'Seattle')] = 100
connections[('Austin', 'New York')] = 200
connections[('New York', 'Austin')] = 400

# 元组是有序的，
# 因此 ('New York', 'Austin') 和 ('Austin', 'New York') 是两个不同的键：
print(connections[('Austin', 'New York')])  # 200
print(connections[('New York', 'Austin')])  # 400

# 字典方法
# get 方法 : d.get(key, default = None)
# 之前已经见过，用索引可以找到一个键对应的值，
# 但是当字典中没有这个键的时候，Python会报错
a = {'first': 'num 1', 'second': 'num 2'}
# error:
# print(a['third'])
# get 返回字典中键 key 对应的值，
# 如果没有这个键，返回 default 指定的值（默认是 None ）。
print(a.get('third'))  # None

# 指定默认值参数：
b = a.get("three", "num 0")
print(b)  # num 0

# pop 方法删除元素
# pop 方法可以用来弹出字典中某个键对应的值，同时也可以指定默认参数：
# d.pop(key, default = None)
a = {'first': 'num 1', 'second': 'num 2'}
c = a.pop('first')
print(c)  # num 1
print(a)  # {u'second': u'num 2'}

# 弹出不存在的键值：
d = a.pop("third", 'not exist')
print(d)  # not exist

# 与列表一样，del 函数可以用来删除字典中特定的键值对，例如：
a = {'first': 'num 1', 'second': 'num 2'}
del a["first"]
print(a)  # {u'second': u'num 2'}

# update方法更新字典
# 之前已经知道，可以通过索引来插入、修改单个键值对，
# 但是如果想对多个键值对进行操作，这种方法就显得比较麻烦，好在有 update 方法：
my_dict = dict([('name', 'lili'),
                ('sex', 'female'),
                ('age', 32),
                ('address', 'beijing')])
# 把 ‘lili' 改成 'lucy'，同时插入 'single' 到 'marriage'
dict_update = {'name': 'lucy', 'marriage': 'single'}
my_dict.update(dict_update)
print(my_dict)
# {u'marriage': u'single',
# u'name': u'lucy',
# u'address': u'beijing',
# u'age': 32,
# u'sex': u'female'}
pprint.pprint(my_dict)  # 华丽丽的显示方式

# in查询字典中是否有该键
barn = {'cows': 1, 'dogs': 5, 'cats': 3}
# in 可以用来判断字典中是否有某个特定的键：
print('chickens' in barn)  # False
print('cows' in barn)  # True

# keys 方法，values 方法和items 方法
# `d.keys()`

# 返回一个由所有键组成的列表；
# `d.values()`

# 返回一个由所有值组成的列表；
# `d.items()`

# 返回一个由所有键值对元组组成的列表；
print(barn.keys())  # [u'cows', u'cats', u'dogs']
print(barn.values())  # [1, 3, 5]
print(barn.items())  # [(u'cows', 1), (u'cats', 3), (u'dogs', 5)]
for key, val in barn.items():
    print(key, val)
    # cows 1
    # cats 3
    # dogs 5
