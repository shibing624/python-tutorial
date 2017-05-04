# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import time

import nltk
from nltk.corpus import brown

start = time.time()
print('start', start)
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist.update(word[-1:])
    suffix_fdist.update(word[-2:])
    suffix_fdist.update(word[-3:])
common_suffixes = list(suffix_fdist.keys())[:100]
print(common_suffixes)


def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith(%s' % suffix] = word.lower().endswith(suffix)
    return features


tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n, g) in tagged_words]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.DecisionTreeClassifier.train(train_set)
suffix_rate = nltk.classify.accuracy(classifier, test_set)
print('suffix_rate', suffix_rate)
end = time.time()
print('end', end)
print(end - start)
print(classifier.pseudocode(depth=4))
