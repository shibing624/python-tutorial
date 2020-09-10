# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from annoy import AnnoyIndex

a = AnnoyIndex(3, 'angular')
a.add_item(0, [1, 0, 0])
a.add_item(1, [0, 1, 0])
a.add_item(2, [0, 0, 1])
a.build(-1)

print(a.get_nns_by_item(0, 100))
print(a.get_nns_by_vector([1.0, 0.5, 0.5], 100))

import random

f = 40
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 100)) # will find the 1000 nearest neighbors