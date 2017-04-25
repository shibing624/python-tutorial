# -*- coding: utf-8 -*-
"""
@description: 
@author:XuMing
"""
from __future__ import print_function  # 兼容python3的print写法
from __future__ import unicode_literals  # 兼容python3的编码处理

import logging

import gensim

finance_txt_path = 'data/C000008.txt'
sentences = open(finance_txt_path, 'r', encoding='utf-8').read().split()
print(sentences[:10])

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(sentences, min_count=1)
model.init_sims(replace=True)
model.save('C000008.word2vec.model')
print('save model ok.')
print(model)
print('')
# # word vector
print(model['中'])
print(model['国'])
#
# # compare two word
print(model.similarity('中', '国'))
