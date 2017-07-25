# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

# 在 Python 中，函数是一种基本类型的对象，这意味着
#
# 可以将函数作为参数传给另一个函数
# 将函数作为字典的值储存
# 将函数作为另一个函数的返回值
def square(x):
    """Square of x."""
    return x * x


def cube(x):
    """Cube of x."""
    return x * x * x


# 函数的基本类型
funcs = {'square': square, 'cube': cube, }
x = 2
for func in sorted(funcs):
    print(func, funcs[func](x))


# 函数参数
# 引用传递

# 传递给函数 f 的是一个指向 x 所包含内容的引用，
# 如果我们修改了这个引用所指向内容的值（例如 x[0]=999），
# 那么外面的 x 的值也会被改变：
def mod_f(x):
    x[0] = 999
    return x


x = [1, 2, 3]

print(x)
print(mod_f(x))
print(x)


# [1, 2, 3]
# [999, 2, 3]
# [999, 2, 3]


# 过如果我们在函数中赋给 x 一个新的值（例如另一个列表），
# 那么在函数外面的 x 的值不会改变：
def no_mod_f(x):
    x = [4, 5, 6]
    return x


x = [1, 2, 3]

print(x)
print(mod_f(x))
print(x)
# [1, 2, 3]
# [4, 5, 6]
# [1, 2, 3]

# 高阶函数
# 以函数作为参数，或者返回一个函数的函数是高阶函数，
# 常用的例子有 map 和 filter 函数

# map(f, sq) 函数将 f 作用到 sq 的每个元素上去，并返回结果组成的列表，
# 相当于：
# [f(s) for s in sq]

print(map(square, range(5)))  # [0, 1, 4, 9, 16]


def is_even(x):
    return x % 2 == 0


print(filter(is_even, range(5)))  # [0, 2, 4]
print(map(square, filter(is_even, range(5))))  # [0, 4, 16]


# reduce(f, sq) 函数接受一个二元操作函数 f(x,y)，
# 并对于序列 sq 每次合并两个元素：
def my_add(x, y):
    return x + y


reduce(my_add, [1, 2, 3])


# 返回一个函数：
def get_logger_func(target):
    def write_logger(data):
        with open(target, 'a') as f:
            f.write(data + '\n')

    return write_logger


fun_logger = get_logger_func('foo.txt')
fun_logger('hello')

# 匿名函数lambda
print(map(square, range(5)))
# 用匿名函数替换为：
print(map(lambda x: x * x, range(5)))

# 匿名函数虽然写起来比较方便（省去了定义函数的烦恼），
# 但是有时候会比较难于阅读：
s1 = reduce(lambda x, y: x + y, map(lambda x: x ** 2, range(1, 3)))
print(s1)  # 5

# 简单的写法：
s2 = sum(x ** 2 for x in range(1, 3))
print(s2)  # 5

# global 变量
# 要在函数中修改全局变量的值，需要加上 global 关键字：
x = 15


def print_newx():
    global x
    x = 18
    print(x)


print_newx()
print(x)
# 18
# 18

# 如果不加上这句 global 那么全局变量的值不会改变：
x = 15


def print_newx2():
    x = 18
    print(x)


print_newx2()
print(x)


# 18
# 15

# 递归
# 一般对于分治法，要用递归，不过在python中不怎么用，
# 更高效的处理非波切利算法：
def fib(n):
    """Fib without recursion."""
    a, b = 0, 1
    for i in range(1, n + 1):
        a, b = b, a + b
    return b


print([fib(i) for i in range(10)])
