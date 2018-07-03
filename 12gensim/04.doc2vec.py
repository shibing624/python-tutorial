# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理
from gensim import utils
import gensim.models.doc2vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import gensim
import sys
import numpy as np
from gensim import corpora, models
import csv
import _pickle as cPickle
from sklearn.externals import joblib
import bz2
from random import shuffle
import ast,os
from sklearn.linear_model import LogisticRegression

# 获取训练与测试数据及其类别标注
neg_file = '../data/douban_imdb_data/neg.txt'
pos_file = '../data/douban_imdb_data/pos.txt'
unsup_file = '../data/douban_imdb_data/unsup.txt'
sentences = gensim.models.doc2vec.TaggedLineDocument(neg_file)
model = gensim.models.doc2vec.Doc2Vec(sentences)
model.save('neg.d2v.model')
model = Doc2Vec.load('neg.d2v.model')
sims = model.docvecs.most_similar(9)
print(sims)


print(model.doesnt_match("annoying is this new IMDB rule of requiring".split()))
print(model.doesnt_match(" over was the fact that ".split()))
print(model.doesnt_match("my god this really".split()))
print(model.doesnt_match("I'm sure I missed some plot points".split()))


print(model.most_similar(positive=['blue', 'shirt'], negative=['blue']))



print(model.most_similar_cosmul(positive=['blue', 'coat'], negative=['yellow']) )