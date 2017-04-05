# -*- coding: utf-8 -*-
"""
@description: 动态编译
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

# Byte Code 编译
# Python, Java 等语言先将代码编译为 byte code（不是机器码），然后再处理：
# .py -> .pyc -> interpreter

# eval(statement, glob, local)
# 使用 eval 函数动态执行代码，返回执行的值：
a = 1
b = eval('a+2')
print(b)

# exec(statement, glob, local)
# 使用 exec 可以添加修改原有的变量:
a = 1
exec ('b = a + 10')
print(b)

local = dict(a=2)
glob = {}
exec ("b = a+1", glob, local)

print(local)

# compile 函数生成 byte code
# compile(str, filename, mode)
a = 1
b = compile('a+2', '', 'eval')
print(eval(b))

a = 1
c = compile("b=a+4", "", 'exec')
exec (c)
print(b)

# abstract syntax trees
import ast

tree = ast.parse('a+10', '', 'eval')
ast.dump(tree)

a = 1
c = compile(tree, '', 'eval')
d = eval(c)
print(d)

# 安全的使用方法 literal_eval ，只支持基本值的操作：
b = ast.literal_eval('[10.0, 2, True, "foo"]')
print(b)
