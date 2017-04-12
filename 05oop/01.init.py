# -*- coding: utf-8 -*-
"""
@description: 特殊方法
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


# 特殊方法
# Python 使用 __ 开头的名字来定义特殊的方法和属性，它们有：
#
# __init__()
# __repr__()
# __str__()
# __call__()
# __iter__()
# __add__()
# __sub__()
# __mul__()
# __rmul__()
# __class__
# __name__

# 构造方法 __init__()
# 在产生对象之后，我们可以向对象中添加属性。
# 事实上，还可以通过构造方法，在构造对象的时候直接添加属性：
class Clothes(object):
    """
    init_demo
    """

    def __init__(self, color="green"):
        self.color = color


my_clothes = Clothes()
print(my_clothes.color)

# 传入有参数的值：
your_clothes = Clothes('orange')
print(your_clothes.color)


# 表示方法 __repr__() 和 __str__()
class Clothes(object):
    """
    repr and str demo
    """

    def __init__(self, color="green"):
        self.color = color

    def __str__(self):
        "This is a string to print."
        return ("a {} clothes".format(self.color))

    def __repr__(self):
        "This string recreates the object."
        return ("{}(color='{}')".format(self.__class__.__name__, self.color))


# __str__() 是使用 print 函数显示的结果,类似java中的toString：
my_clothes = Clothes()
print(my_clothes)

# __repr__() 返回的是不使用 print 方法的结果:
print(my_clothes.__class__, my_clothes.__class__.__name__, my_clothes.color)
