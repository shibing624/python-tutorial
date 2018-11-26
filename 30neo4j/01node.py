# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from py2neo import Node,Relationship

a = Node('Person',name='Alice')
b = Node('Person',name='Bob')
r = Relationship(a,"KNOWS",b)
print(a,b,r)

s = a|b|r
print(s)

