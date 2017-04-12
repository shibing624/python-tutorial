# -*- coding: utf-8 -*-
"""
@description: 接口
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


class Clothes(object):
    def __init__(self, color="green"):
        self.color = color

    def out(self):
        print("father.")


class NikeClothes(Clothes):
    def out(self):
        self.color = "brown"
        super(NikeClothes, self).out()


class AdidasClothes(object):
    def out(self):
        print("adidas.")


# 因为三个类都实现了 out() 方法，因此可以这样使用：
objects = [Clothes(), NikeClothes(), AdidasClothes()]
for obj in objects:
    obj.out()
