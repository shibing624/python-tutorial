# -*- coding: utf-8 -*-

"""
@author: XuMing <shibing624@126.com>
@summary:
"""

import os
import jieba
f = open('test.txt.txt',encoding='utf8').read()
print(f)
text = open('../data/tianlongbabu.txt',encoding='utf8').read()
text = ' '.join(jieba.lcut(f))
print(text)