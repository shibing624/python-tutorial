# -*- coding: utf-8 -*-
"""
@description: 介绍pickle的使用方法
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

import os

# cPickle 使用 C 而不是 Python 实现了相同的算法，因此速度上要比 pickle 快一些。
# 但是它不允许用户从 pickle 派生子类。如果子类对你的使用来说无关紧要，
# 那么 cPickle 是个更好的选择
try:
    import cPickle as pickle
except:
    import pickle

data = [{'a': 'A', 'b': 2, 'c': 2.22}]
# 使用 pickle.dumps() 可以将一个对象转换为字符串（dump string）：
data_string = pickle.dumps(data)
print("DATA:")
print(data)
print("PICKLE:")
print(data_string)

# 虽然 pickle 编码的字符串并不一定可读，但是我们可以
# 用 pickle.loads() 来从这个字符串中恢复原对象中的内容（load string）：
data_from_string = pickle.loads(data_string)
print(data_from_string)

# dumps 可以接受一个可省略的 protocol 参数（默认为 0）
data_string_0 = pickle.dumps(data, 0)

print("Pickle 0:", data_string_0)

data_string_1 = pickle.dumps(data, 1)

print("Pickle 1:", data_string_1)

data_string_2 = pickle.dumps(data, 2)

print("Pickle 2:", data_string_2)
# 如果 protocol 参数指定为负数，那么将调用当前的最高级的编码协议进行编码：
print(pickle.dumps(data, -1))
# 从这些格式中恢复对象时，不需要指定所用的协议，pickle.load() 会自动识别：
print("Load 1:", pickle.loads(data_string_1))
print("Load 2:", pickle.loads(data_string_2))

# 存储和读取 pickle 文件
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('data.pkl',"rb") as f:
    data_from_file = pickle.load(f)

print(data_from_file)

# 清理生成的文件：
os.remove('data.pkl')
