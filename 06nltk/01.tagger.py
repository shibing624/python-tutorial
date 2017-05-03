# -*- coding: utf-8 -*-
"""
@description: tagger
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk

# 默认字典
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = list(vocab)[:1000]
mapping = nltk.defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v

alice2 = [mapping[v] for v in alice]
print(alice2[:100])
print(len(set(alice2)))

# 反向查找
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV', 'old': 'ADJ', 'search': 'V'}
pos2 = nltk.Index((value, key) for (key, value) in pos.items())
adjs = pos2['ADJ']
print(adjs)
advs = pos2['ADV']
print(advs)

# 默认标注器
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
max = nltk.FreqDist(tags).max()
print(max)

# build a pos of NN
raw = 'i do not like green color , and i do not like them, Sam i am'
tokens = nltk.word_tokenize(raw)
print('tokens', tokens)
print('brown_tagged_tokens', brown_tagged_sents)
default_tagger = nltk.DefaultTagger('NN')
new_tokens = default_tagger.tag(tokens)
print('new_tokens', new_tokens)

# NN in brown tate:
rate = default_tagger.evaluate(brown_tagged_sents)
print(rate)

# regex pattern
patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*', 'NN')
]
regexp_tagger = nltk.RegexpTagger(patterns)
tagged = regexp_tagger.tag(brown_sents[3])
print(tagged)
regex_rate = regexp_tagger.evaluate(brown_tagged_sents)
print('regex_rate', regex_rate)

# select tagger
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = list(fd.keys())[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
baseline_rate = baseline_tagger.evaluate(brown_tagged_sents)
print('baseline_rate', baseline_rate)

sent = brown.sents(categories='news')[3]
print(baseline_tagger.tag(sent))
