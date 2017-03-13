# -*- coding: utf-8 -*-
"""
@description: 介绍字符串的索引及切分操作
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

s = 'good morning'
print(s[0])  # g

print(s[-2])  # n

# 分片用来从序列中提取出想要的子序列，其用法为：

# var[lower:upper:step]

# 其范围包括 lower ，但不包括 upper ，即 [lower, upper)，
# step 表示取值间隔大小，如果没有默认为1。
print(s[-3:])  # ing
print(s[:-3])  # good morn
print(s[:])  # good morning

print(s[::2])  # go onn
print(s[::-1])  # gninrom doog
print(s[:100])
