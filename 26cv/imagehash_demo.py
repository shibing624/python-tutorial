# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import imagehash
from PIL import Image

h = imagehash.average_hash(Image.open('./search/data/images/car-1.jpg'))
print(h)

o = imagehash.average_hash(Image.open('./search/data/images/car-2.jpg'))
print(o)

print(h == o)
print(h - o)

print('-' * 44)
h = imagehash.average_hash(Image.open('./search/data/images/orig.jpg'))
print(h)

o = imagehash.average_hash(Image.open('./search/data/images/mod.jpg'))
print(o)

print(h == o)
print(h - o)

print('-' * 44)
h = imagehash.phash(Image.open('./search/data/images/orig.jpg'))
print(h)

o = imagehash.phash(Image.open('./search/data/images/mod.jpg'))
print(o)

print(h == o)
print(h - o)
