# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 正则表达式示例
"""

import re

a = "物流太糟糕了，申通{服务恶劣啊，自己去取的"
for i in re.finditer('，', a):
    print(i.group(), i.span()[1])

s = '{通配符}你好，今天开学了{通配符},你好'
print("s", s)

a1 = re.compile(r'\{.*?\}')
d = a1.sub('', s)
print("d", d)

a1 = re.compile(r'\{[^}]*\}')
d = a1.sub('', s)
print("d", d)
