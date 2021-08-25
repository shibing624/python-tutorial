# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""


import nltk
sentence= """At eight o'clock on Thursday morning 
Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)

from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()

