# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

size = int(len(brown_tagged_sents) * 0.9)
print(size)  # 4160

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# 一元标注器
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_rate = unigram_tagger.evaluate(test_sents)
print('unigram_rate', unigram_rate)  # 0.8121200039868434

# 二元标注器
bigram_tagger = nltk.BigramTagger(train_sents)
print(bigram_tagger.tag(brown_sents[2007]))
unseen_sent = brown_sents[4203]
print(bigram_tagger.tag(unseen_sent))
bigram_rate = bigram_tagger.evaluate(test_sents)
print('bigram_rate', bigram_rate)

# 组合 bigram 标注器、unigram 标注器和一个默认标注器
# 1. 尝试使用 bigram 标注器标注标识符。
# 2. 如果 bigram 标注器无法找到一个标记，尝试 unigram 标注器。
# 3. 如果 unigram 标注器也无法找到一个标记，使用默认标注器。
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t3 = nltk.TrigramTagger(train_sents, cutoff=2, backoff=t2)
conbine_rate = t3.evaluate(test_sents)
print('conbine_rate', conbine_rate)

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
conbine_rate_simple = t2.evaluate(test_sents)
print('conbine_rate_simple', conbine_rate_simple)
