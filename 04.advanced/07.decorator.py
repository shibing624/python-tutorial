# -*- coding: utf-8 -*-
"""
@description: 装饰器
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


# 用 @ 来使用装饰器
# 使用 @ 符号来将某个函数替换为装饰符之后的函数：

# 例如这个函数：
def dec(f):
    print('I am decorating function', id(f))
    return f


def foo(x):
    print(x)  # I am decorating function 45206384


foo = dec(foo)


# 可以替换为：
@dec
def foo(x):
    print(x)


# 例子
# 定义两个装饰器函数，一个将原来的函数值加一，另一个乘二：
def plus_one(f):
    def new_func(x):
        return f(x) + 1

    return new_func


def times_two(f):
    def new_func(x):
        return f(x) * 2

    return new_func


# 定义函数，先乘二再加一：
@plus_one
@times_two
def foo(x):
    return int(x)


b = foo(2)
print(b)  # 5


# 修饰器工厂
# decorators factories 是返回修饰器的函数

# 它的作用在于产生一个可以接受参数的修饰器，
# 例如我们想将 函数 输出的内容写入一个文件去，可以这样做：
def super_loud(filename):
    fp = open(filename, 'w')

    def loud(f):
        def new_func(*args, **kw):
            fp.write(str(args))
            fp.writelines('\n')
            fp.write('calling with' + str(args) + str(kw))
            # 确保内容被写入
            fp.flush()
            fp.close()
            rtn = f(*args, **kw)
            return rtn

        return new_func

    return loud


@super_loud('test.txt')
def foo(x):
    print(x)


# 调用 foo 就会在文件中写入内容：
foo(100)

# 查看文件：
with open('test.txt') as f:
    print(f.read())


# import os
# os.remove('test.txt')

# @classmethod 装饰器
# 在 Python 标准库中，有很多自带的装饰器，
# 例如 classmethod 将一个对象方法转换了类方法：
class Foo(object):
    @classmethod
    def bar(cls, x):
        print('the input is', x)

    def __init__(self):
        pass


# 类方法可以通过 类名.方法 来调用：
Foo.bar(10)


# @property 装饰器
# 有时候，我们希望像 Java 一样支持 getters 和 setters 的方法，
# 这时候就可以使用 property 装饰器：
class Foo(object):
    def __init__(self, data):
        self.data = data

    @property
    def x(self):
        return self.data


# 此时可以使用 .x 这个属性查看数据（不需要加上括号）：
foo = Foo(22)
print(foo.x)


# 这样做的好处在于，这个属性是只读的：
# foo.x = 1 会报错

# 如果想让它变成可读写，可以加上一个装饰符 @x.setter：
class Foo(object):
    def __init__(self, data):
        self.data = data

    @property
    def x(self):
        return self.data

    @x.setter
    def x(self, value):
        self.data = value


foo = Foo(1000)
print(foo.x)
foo.x = 2222
print(foo.x)
