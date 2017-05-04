# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import math

import nltk


def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in nltk.FreqDist(labels)]
    return -sum([p * math.log(p, 2) for p in probs])


print(entropy(['male', 'male', 'male', 'male']))
print(entropy(['male', 'male', 'male', 'female']))
print(entropy(['male', 'female', 'male', 'female']))
print(entropy(['female', 'female', 'male', 'female']))

# 我们选择了分布 i 因为它的标签概率分布均匀——换句话说，因为它的熵较高。
# 一般情况下， 最大熵原理是说在与我们所知道的一致的的分布中，我们会选择熵最高的。
