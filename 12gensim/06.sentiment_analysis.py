# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

from random import shuffle

from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


sources = {'/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/neg_train.txt': 'TRAIN_NEG',
           '/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/pos_train.txt': 'TRAIN_POS',
           '/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/uns_train.txt': 'TRAIN_UNS',
           '/Volumes/Macintosh HD/Users/RayChou/Downloads/情感分析训练语料/uns_test.txt': 'TEST_UNS'}
sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=15, size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())
