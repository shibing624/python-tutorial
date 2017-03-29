# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# Numpy 有对内存映射的支持。

# 内存映射也是一种处理文件的方法，主要的函数有：
#
# memmap
# frombuffer
# ndarray constructor

# 使用内存映射文件处理存储于磁盘上的文件时，将不必再对文件执行I/O操作，
# 使得内存映射文件在处理大数据量的文件时能起到相当重要的作用。

# memmap(filename,
#        dtype=uint8,
#        mode='r+'
#        offset=0
#        shape=None
#        order=0)

# mode 表示文件被打开的类型：
# r 只读
# c 复制+写，但是不改变源文件
# r+ 读写，使用 flush 方法会将更改的内容写入文件
# w+ 写，如果存在则将数据覆盖
