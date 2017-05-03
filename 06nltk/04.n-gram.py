# -*- coding: utf-8 -*-
"""
@description: n-gram
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理
import nltk
import pickle
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

size = int(len(brown_tagged_sents) * 0.9)
print(size)  # 4160

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)

output = open('t2.pkl','wb')
pickle.dump(t2,output,-1)
output.close()

input = open('t2.pkl','rb')
tagger = pickle.load(input)