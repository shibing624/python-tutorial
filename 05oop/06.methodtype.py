# -*- coding: utf-8 -*-
"""
@description: 共有，私有和特殊方法和属性
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理


#  special 方法和属性，即以 __ 开头和结尾的方法和属性
# 私有方法和属性，以 _ 开头，不过不是真正私有，而是可以调用的，
# 但是不会被代码自动完成所记录（即 Tab 键之后不会显示）
# 其他都是共有的方法和属性
# 以 __ 开头不以 __ 结尾的属性是更加特殊的方法，调用方式也不同：
class MyDemoClass(object):
    def __init__(self):
        print("special.")

    def _get_name(self):
        print("_get_name is private method.")

    def get_value(self):
        print("get_value is public method.")

    def __get_type(self):
        print("__get_type is really special method.")


demo = MyDemoClass()

demo.get_value()
demo._get_name()
demo._MyDemoClass__get_type()
