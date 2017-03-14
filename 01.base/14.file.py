# -*- coding: utf-8 -*-
"""
@description:
@author:XuMing
"""
from __future__ import print_function
from __future__ import unicode_literals

# 读文件
# 使用 open 函数或者 file 函数来读文件，使用文件名的字符串作为输入参数：
f = open("test.txt")
f = open("test.txt")
# 这两种方法没太大区别

# 默认以读的方式打开文件，如果文件不存在会报错。
# 可以使用 read 方法来读入文件中的所有内容：
text = f.read()
print(text)

# 按照行读入内容，readlines 方法返回一个列表，每个元素代表文件中每一行的内容：
f = open("test.txt")
lines = f.readlines()
print(lines)
f.close()

# 事实上，我们可以将 f 放在一个循环中，得到它每一行的内容：
f = open('test.txt')
for line in f:
    print(line)
f.close()

# 写文件
# 我们使用 open 函数的写入模式来写文件：
f = open('test.txt', 'w')
f.write('hello world.')
f.close()

print(open('test.txt').read())
# 使用 w 模式时，如果文件不存在会被创建
# 除了写入模式，还有追加模式 a

# 读写模式w+
f = open('test.txt', 'w+')
f.write('hello world. morning.')
f.seek(3)
print(f.read())  # hello world.
f.close()

# 二进制文件
# 二进制读写模式 b：
import os

f = open('binary.bin', 'wb')
f.write(os.urandom(10))
f.close()

f = open('binary.bin', 'rb')
print(repr(f.read()))
f.close()

# with 方法
# 事实上，Python提供了更安全的方法，当 with 块的内容结束后，
# Python会自动调用它的close 方法，确保读写的安全：
with open('newfile.txt', 'w') as f:
    for i in range(3000):
        x = 1.0 / (i - 1000)
        f.write('hello world: ' + str(i) + '\n')

# 与 try/exception/finally 效果相同，但更简单。
