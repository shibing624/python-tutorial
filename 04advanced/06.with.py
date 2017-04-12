# -*- coding: utf-8 -*-
"""
@description: with 语句和上下文管理器
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

with open('my_file.txt', 'w') as fp:
    data = fp.write("Hello world")

# 这等效于下面的代码，但是要更简便：
fp = open('my_file.txt', 'w')
try:
    # do stuff with f
    data = fp.write("Hello world")
finally:
    fp.close()


# 比如可以这样定义一个简单的上下文管理器：
class ContextManager(object):
    def __enter__(self):
        print("Entering")

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")


with ContextManager():
    print("inside operate")


# __enter__ 的返回值
# 如果在 __enter__ 方法下添加了返回值，
# 那么我们可以使用 as 把这个返回值传给某个参数：
class ContextManager2(object):
    def __enter__(self):
        print("Entering")
        return "my value"

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")


with ContextManager2() as val:
    print(val)


# 一个通常的做法是将 __enter__ 的返回值设为这个上下文管理器对象本身，
# 文件对象就是这样做的.
class ContextManager3(object):
    def __enter__(self):
        print("Entering")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")


# 错误处理
# 上下文管理器对象将错误处理交给 __exit__ 进行，可以将错误类型，
# 错误值和 traceback 等内容作为参数传递给 __exit__ 函数：
class ContextManager4(object):
    def __enter__(self):
        print("Entering")

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")
        if exc_type is not None:
            print("  Exception:", exc_value)
            return True  # 不想让错误抛出，只需要将 __exit__ 的返回值设为 True


with ContextManager4():
    print(1 / 0)
