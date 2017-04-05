# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

# 与操作系统进行交互：os 模块
import os

# 文件路径操作
# os.remove(path) 或 os.unlink(path) ：删除指定路径的文件。路径可以是全名，也可以是当前工作目录下的路径。
# os.removedirs：删除文件，并删除中间路径中的空文件夹
# os.chdir(path)：将当前工作目录改变为指定的路径
# os.getcwd()：返回当前的工作目录
# os.curdir：表示当前目录的符号
# os.rename(old, new)：重命名文件
# os.renames(old, new)：重命名文件，如果中间路径的文件夹不存在，则创建文件夹
# os.listdir(path)：返回给定目录下的所有文件夹和文件名，不包括 '.' 和 '..' 以及子文件夹下的目录。（'.' 和 '..' 分别指当前目录和父目录）
# os.mkdir(name)：产生新文件夹
# os.makedirs(name)：产生新文件夹，如果中间路径的文件夹不存在，则创建文件夹

# 产生文件：
f = open('test.file', 'w')
f.close()
print('test.file' in os.listdir(os.curdir))

# 重命名文件
os.rename("test.file", "test.new.file")
print("test.file" in os.listdir(os.curdir))
print("test.new.file" in os.listdir(os.curdir))

# 删除文件
os.remove("test.new.file")

# 系统常量
# windows 为 \r\n
# unix为 \n
print(os.linesep)
# 当前操作系统的路径分隔符：
print(os.sep)
# 当前操作系统的环境变量中的分隔符（';' 或 ':'）：
# windows 为 ;
# unix 为:
print(os.pathsep)

# os.environ 是一个存储所有环境变量的值的字典，可以修改。
print(os.environ)

# os.path 模块
import os.path

# os.path.isfile(path) ：检测一个路径是否为普通文件
# os.path.isdir(path)：检测一个路径是否为文件夹
# os.path.exists(path)：检测路径是否存在
# os.path.isabs(path)：检测路径是否为绝对路径
print(os.path.isfile("C:/Windows"))
print(os.path.isdir("C:/Windows"))
print(os.path.exists("C:/Windows"))
print(os.path.isabs("C:/Windows"))
# split 和 join
# os.path.split(path)：拆分一个路径为 (head, tail) 两部分
# os.path.join(a, *p)：使用系统的路径分隔符，将各个部分合成一个路径
head, tail = os.path.split("c:/tem/b.txt")
print(head, tail)
a = "c:/tem"
b = "b.txt"
print(os.path.join(a, b))


def get_files(dir_path):
    '''
    列出文件夹下的所有文件
    :param dir_path: 父文件夹路径
    :return: 
    '''
    for parent, dirname, filenames in os.walk(dir_path):
        for filename in filenames:
            print("parent is:", parent)
            print("filename is:", filename)
            print("full name of the file is:", os.path.join(parent, filename))


dir = "C:\Windows\System32\drivers\etc"
get_files(dir)
