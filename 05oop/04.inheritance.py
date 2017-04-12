# -*- coding: utf-8 -*-
"""
@description: 继承
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


# 类定义的基本形式：
# class ClassName(ParentClass):
#     """class docstring"""
#     def method(self):
#         return

# 里面的 ParentClass 就是用来继承的
class Clothes(object):
    def __init__(self, color="green"):
        self.color = color

    def out_print(self):
        return self.__class__.__name__, self.color


# Test:
my_clothes = Clothes()
print(my_clothes.color)
print(my_clothes.out_print())


# 定义一个子类，继承父类的所有方法
class NikeClothes(Clothes):
    def change_color(self):
        if self.color == "green":
            self.color = "red"


# 继承父类的所有方法：
your_clothes = NikeClothes()
print(your_clothes.color)
print(your_clothes.out_print())
# 但有自己的方法
your_clothes.change_color()
print(your_clothes.color)


# 如果想对父类的方法进行修改，只需要在子类中重定义这个类即可：
class AdidasClothes(Clothes):
    def change_color(self):
        if self.color == "green":
            self.color = "black"

    def out_print(self):
        self.change_color()
        return self.__class__.__name__, self.color


him_clothes = AdidasClothes()
print(him_clothes.color)
him_clothes.change_color()
print(him_clothes.color)
print(him_clothes.out_print())


# super() 函数
# super(CurrentClassName, instance)
#
# 返回该类实例对应的父类对象。
# 刚才 AdidasClothes可以改写为：
class NewAdidasClothes(Clothes):
    def change_color(self):
        if self.color == "green":
            self.color = "black"

    def out_print(self):
        self.change_color()
        print(super(NewAdidasClothes, self).out_print())


# Test
her_clothes = NewAdidasClothes()
print(her_clothes.color)
her_clothes.out_print()
