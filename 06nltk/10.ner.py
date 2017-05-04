# -*- coding: utf-8 -*-
"""
@description: 命名实体识别
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import nltk


def ie_preprocess(doc):
    sentences = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]


sentence = [('the', 'DT'), ('little', 'JJ'), ('yellow', 'JJ'),
            ('dog', 'NN'), ('barked', 'VBD'), ('at', 'IN'), ('the', 'DT'), ('cat', 'NN')]
grammar = 'NP:{<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw()

sentence = [('another','DT'),('sharp','JJ'),('dive','NN'),('trade','NN'),
            ('figures','NNS'),('any','DT'),('new','JJ'),('policy','NN'),
            ('measures','NNS'),('earlier','JJR'),('stages','NNS'),('Panamanian','JJ'),
            ('dictator','NN'),('Manuel','NNP'),('Noriega','NNP')]
grammar = 'NP:{<DT>?<JJ.*>*<NN.*>+}'
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw()

