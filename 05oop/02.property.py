# -*- coding: utf-8 -*-
"""
@description: 属性
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


# 只读属性
class Clothes(object):
    def __init__(self, price):
        self.price = price

    # 这样 discount_price 就变成属性了
    @property
    def discount_price(self):
        return self.price * 0.8


# 这里 discount_price 就是一个只读不写的属性了（注意是属性不是方法）,
# 而price是可读写的属性：
my_clothes = Clothes(100)
print(my_clothes.discount_price)  # 80.0

# 可以修改price属性来改变discount_price：
my_clothes.price = 200
print(my_clothes.discount_price)  # 160.0


# my_clothes.discount_price()会报错，因为 my_clothes.discount_price 是属性，不是方法；
# my_clothes.discount_price=100 也会报错，因为只读

# 对于 @property 生成的只读属性，
# 我们可以使用相应的 @attr.setter 修饰符来使得这个属性变成可写的：
class Clothes(object):
    def __init__(self, price):
        self.price = price

    # 这样就变成属性了
    @property
    def discount_price(self):
        return self.price * 0.8

    @discount_price.setter
    def discount_price(self, new_price):
        self.price = new_price * 1.25


# example:
my_clothes = Clothes(100)
print(my_clothes.discount_price)

my_clothes.price = 200
print(my_clothes.discount_price)

# 修改 discount_price 属性：
my_clothes.discount_price = 180
print(my_clothes.price)
print(my_clothes.discount_price)


# 一个等价的替代如下，用方法：
class Clothes(object):
    def __init__(self, price):
        self.price = price

    def get_discount_price(self):
        return self.price * 0.8

    def set_discount_price(self, new_price):
        self.price = new_price * 1.25

    discount_price = property(get_discount_price, set_discount_price)
